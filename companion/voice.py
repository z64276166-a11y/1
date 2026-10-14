"""耳朵（faster-whisper 本地识别）和嘴巴（edge-tts / GPT-SoVITS）。"""
import asyncio
import os

import httpx
import numpy as np


def _has_cuda() -> bool:
    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


class Ears:
    def __init__(self):
        from faster_whisper import WhisperModel

        device = os.getenv("WHISPER_DEVICE", "auto")
        if device == "auto":
            device = "cuda" if _has_cuda() else "cpu"
        model = os.getenv("WHISPER_MODEL", "auto")
        if model == "auto":
            model = "large-v3" if device == "cuda" else "small"
        compute = os.getenv("WHISPER_COMPUTE", "auto")
        if compute == "auto":
            compute = "float16" if device == "cuda" else "int8"
        print(f"[耳朵] faster-whisper {model} on {device} ({compute})，首次运行会自动下载模型…")
        self.model = WhisperModel(model, device=device, compute_type=compute)
        self.desc = f"{model}/{device}"

    def _run(self, pcm: np.ndarray) -> str:
        segments, _ = self.model.transcribe(
            pcm, language="zh", beam_size=5, vad_filter=True,
            initial_prompt="以下是普通话的句子，使用简体中文。")
        return "".join(s.text for s in segments).strip()

    async def transcribe(self, pcm: np.ndarray) -> str:
        return await asyncio.to_thread(self._run, pcm)


class Mouth:
    def __init__(self):
        self.engine = os.getenv("TTS_ENGINE", "edge")
        self.http = httpx.AsyncClient(timeout=60)

    async def speak(self, text: str) -> tuple[bytes, str]:
        """返回 (音频字节, mime)。"""
        if self.engine == "gptsovits":
            params = {
                "text": text, "text_lang": "zh",
                "ref_audio_path": os.getenv("GPTSOVITS_REF_AUDIO"),
                "prompt_text": os.getenv("GPTSOVITS_PROMPT_TEXT", ""),
                "prompt_lang": os.getenv("GPTSOVITS_PROMPT_LANG", "zh"),
                "media_type": "wav", "streaming_mode": "false",
            }
            r = await self.http.get(os.getenv("GPTSOVITS_URL", "http://127.0.0.1:9880/tts"), params=params)
            r.raise_for_status()
            return r.content, "audio/wav"

        import edge_tts
        comm = edge_tts.Communicate(
            text, os.getenv("EDGE_VOICE", "zh-CN-XiaoyiNeural"),
            rate=os.getenv("EDGE_RATE", "+5%"), pitch=os.getenv("EDGE_PITCH", "+0Hz"))
        buf = bytearray()
        async for chunk in comm.stream():
            if chunk["type"] == "audio":
                buf += chunk["data"]
        return bytes(buf), "audio/mpeg"
