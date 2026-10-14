"""数字人女友 - 服务端。启动后浏览器打开 http://127.0.0.1:8765"""
import asyncio
import base64
import json
import os
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")

from fastapi import FastAPI, WebSocket, WebSocketDisconnect  # noqa: E402
from fastapi.responses import FileResponse, PlainTextResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from brain import Brain, SentenceSplitter, clean_for_speech  # noqa: E402
from memory import Memory  # noqa: E402
from voice import Ears, Mouth  # noqa: E402

app = FastAPI()
memory = Memory()
brain = Brain(memory)
mouth = Mouth()
ears: Ears | None = None
consolidate_lock = asyncio.Lock()
AVATAR_MODE = os.getenv("AVATAR_MODE", "live2d")  # live2d | realistic
LT_URL = os.getenv("LIVETALKING_URL", "http://127.0.0.1:8010").rstrip("/")
DEFAULT_MODEL = ("https://cdn.jsdelivr.net/gh/guansss/pixi-live2d-display/test/assets/"
                 "haru/haru_greeter_t03.model3.json")


@app.on_event("startup")
async def _load_models():
    global ears
    try:
        ears = await asyncio.to_thread(Ears)
    except Exception as e:  # 没装好也能用文字聊
        print(f"[耳朵] 语音识别加载失败，只能打字聊天：{e}")


@app.get("/")
async def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/config")
async def config():
    return {
        "name": brain.name,
        "idleSeconds": int(os.getenv("IDLE_SECONDS", "60")),
        "asr": ears.desc if ears else None,
        "model": os.getenv("LIVE2D_MODEL") or DEFAULT_MODEL,
        "avatarMode": AVATAR_MODE,
        "liveTalkingUrl": LT_URL,
    }


@app.get("/memory", response_class=PlainTextResponse)
async def show_memory():
    return memory.export_json()


app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


async def run_consolidate():
    if consolidate_lock.locked():
        return
    async with consolidate_lock:
        try:
            await brain.consolidate()
        except Exception as e:
            print(f"[记忆] 整理失败：{e}")


class Call:
    """一次视频通话。新的输入会打断正在进行的回复。"""

    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.task: asyncio.Task | None = None
        self.turn = 0
        self.user_turns = 0
        self.lt_session: str | None = None  # 写实模式下 LiveTalking 的会话

    async def lt_post(self, path: str, **body):
        if not self.lt_session:
            return
        try:
            r = await mouth.http.post(f"{LT_URL}{path}", json={"sessionid": self.lt_session, **body})
            if r.json().get("code") != 0:
                print(f"[形象] LiveTalking {path} 返回：{r.text}")
        except Exception as e:
            print(f"[形象] 连接 LiveTalking 失败：{e}")

    async def send(self, **msg):
        await self.ws.send_text(json.dumps(msg, ensure_ascii=False))

    async def interrupt(self):
        await self.lt_post("/interrupt_talk")
        if self.task and not self.task.done():
            self.task.cancel()
            try:
                await self.task
            except (asyncio.CancelledError, Exception):
                pass

    async def start_reply(self, hidden: str | None = None):
        await self.interrupt()
        self.turn += 1
        self.task = asyncio.create_task(self.reply(self.turn, hidden))

    async def reply(self, turn: int, hidden: str | None):
        full = ""
        splitter = SentenceSplitter()
        queue: asyncio.Queue[str | None] = asyncio.Queue()

        async def tts_worker():
            while (sent := await queue.get()) is not None:
                speech = clean_for_speech(sent)
                if not speech:
                    continue
                if self.lt_session:
                    # 写实模式：让 LiveTalking 合成声音并驱动真人口型，画面通过 WebRTC 推给浏览器
                    await self.lt_post("/human", text=speech, type="echo", interrupt=False)
                    await self.send(type="say", turn=turn, text=speech)
                    continue
                try:
                    audio, mime = await mouth.speak(speech)
                except Exception as e:
                    print(f"[嘴巴] 语音合成失败：{e}")
                    continue
                await self.send(type="audio", turn=turn, text=speech, mime=mime,
                                data=base64.b64encode(audio).decode())

        worker = asyncio.create_task(tts_worker())
        try:
            await self.send(type="reply_start", turn=turn)
            async for delta in brain.stream_reply(hidden):
                full += delta
                await self.send(type="reply_delta", turn=turn, text=delta)
                for s in splitter.feed(delta):
                    queue.put_nowait(s)
            for s in splitter.flush():
                queue.put_nowait(s)
            queue.put_nowait(None)
            await worker
            await self.send(type="reply_end", turn=turn)
        except asyncio.CancelledError:
            worker.cancel()
            if full:
                full += "……"
            raise
        except Exception as e:
            worker.cancel()
            await self.send(type="error", text=f"大脑出错了：{e}")
        finally:
            if full.strip():
                memory.add_message("assistant", clean_for_speech(full) or full)

    async def on_user_text(self, text: str):
        text = text.strip()
        if not text:
            return
        await self.interrupt()
        await self.send(type="user_text", text=text)
        memory.add_message("user", text)
        self.user_turns += 1
        if self.user_turns % 6 == 0:
            asyncio.create_task(run_consolidate())
        await self.start_reply()


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    call = Call(ws)
    last = memory.last_chat_time()
    gap = time.time() - last if last else None
    if gap is None:
        opening = "这是你们第一次视频通话，对方刚接通。自然、带点害羞地打个招呼。"
    elif gap < 600:
        opening = "对方刚刚又重新接通了视频，接着刚才的话题自然地说一句。"
    else:
        opening = ("对方刚接通视频。像真正的女朋友那样打招呼，结合现在的时间和隔了多久没聊，"
                   "如果记得有待跟进的事（比如对方之前说要做的事情），可以自然地问一下。")
    idle_count = 0
    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if msg.get("bytes") is not None:
                # 浏览器发来的一句话：16kHz float32 PCM
                await call.interrupt()
                idle_count = 0
                if ears is None:
                    await call.send(type="error", text="语音识别没有加载，请看服务端窗口的报错")
                    continue
                pcm = np.frombuffer(msg["bytes"], dtype=np.float32).copy()
                text = await ears.transcribe(pcm)
                if text:
                    await call.on_user_text(text)
                continue
            data = json.loads(msg.get("text") or "{}")
            kind = data.get("type")
            if kind == "hello":
                call.lt_session = data.get("avatarSession") or None
                await call.start_reply(opening)
            elif kind == "text":
                idle_count = 0
                await call.on_user_text(data.get("text", ""))
            elif kind == "interrupt":
                await call.interrupt()
            elif kind == "idle" and idle_count < 2:
                idle_count += 1
                await call.start_reply("对方已经安静了一会儿没说话（可能在忙或者在发呆）。"
                                 "自然地找个话题，或者分享你自己的事，或者关心一下对方。不要说'你怎么不说话'这种话。")
    except WebSocketDisconnect:
        pass
    finally:
        await call.interrupt()
        asyncio.create_task(run_consolidate())


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8765"))
    print(f"\n  打开浏览器访问：http://127.0.0.1:{port}\n")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
