"""大脑：DeepSeek 对话 + 记忆整理。"""
from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

from openai import AsyncOpenAI

from memory import Memory

WEEKDAYS = "一二三四五六日"


def _fmt_gap(seconds: float) -> str:
    if seconds < 600:
        return "刚刚还在聊"
    if seconds < 3600:
        return f"{int(seconds // 60)} 分钟"
    if seconds < 86400:
        return f"{int(seconds // 3600)} 小时"
    return f"{int(seconds // 86400)} 天"


def _fmt_ts(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%m月%d日")


class Brain:
    def __init__(self, memory: Memory):
        self.mem = memory
        self.client = AsyncOpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        )
        self.model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        self.name = os.getenv("COMPANION_NAME", "小雪")
        self.user = os.getenv("USER_NAME", "你")
        self.persona_path = Path(__file__).parent / os.getenv("PERSONA_FILE", "persona.md")

    def persona(self) -> str:
        text = self.persona_path.read_text(encoding="utf-8")
        return text.replace("{name}", self.name).replace("{user}", self.user)

    def system_prompt(self, query: str) -> str:
        now = datetime.now()
        parts = [self.persona()]
        parts.append(f"\n## 当前情况\n现在是 {now:%Y年%m月%d日 %H:%M}，星期{WEEKDAYS[now.weekday()]}。")
        last = self.mem.last_chat_time()
        if last:
            parts.append(f"距离你们上次说话：{_fmt_gap(time.time() - last)}。")
        facts = self.mem.facts()
        if facts:
            parts.append("\n## 你记得的关于对方的事（自然地运用，不要生硬背诵）")
            parts += [f"- {f['content']}" for f in facts]
        eps = self.mem.relevant_episodes(query)
        if eps:
            parts.append("\n## 你们之前聊过的事")
            parts += [f"- [{_fmt_ts(e['ts'])}] {e['summary']}" for e in eps]
        return "\n".join(parts)

    async def stream_reply(self, hidden_instruction: str | None = None):
        """流式生成回复。hidden_instruction 用于主动开口（不存入聊天记录）。"""
        history = self.mem.recent_messages()
        query = " ".join(m["content"] for m in history[-4:] if m["role"] == "user")
        messages = [{"role": "system", "content": self.system_prompt(query)}] + history
        if hidden_instruction:
            messages.append({"role": "user", "content": f"（旁白，对方听不到：{hidden_instruction}）"})
        stream = await self.client.chat.completions.create(
            model=self.model, messages=messages, stream=True, temperature=1.1, max_tokens=300)
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def consolidate(self):
        """把新的聊天内容整理成长期记忆。"""
        rows = self.mem.unprocessed()
        if len(rows) < 2:
            return
        convo = "\n".join(
            f"{'对方' if r['role'] == 'user' else self.name}: {r['content']}" for r in rows)
        facts = self.mem.facts(200)
        facts_txt = "\n".join(f"[{f['id']}] {f['content']}" for f in facts) or "（暂无）"
        prompt = f"""你是{self.name}的记忆系统。根据下面这段新对话，更新关于"对方"的长期记忆。

已有记忆：
{facts_txt}

新对话：
{convo}

请输出 json，格式：
{{"fact_ops": [{{"op": "add|update|delete", "id": 已有记忆编号(update/delete时必填), "content": "一句话事实", "importance": 1-5}}],
 "episode": "用一两句话概括这段对话发生了什么、对方的情绪、有没有约定或待跟进的事；如果只是寒暄则为空字符串"}}

规则：
- 只记录关于对方的稳定信息（身份、喜好、经历、人际关系、计划、雷区）和你们关系中的重要时刻
- 新信息和已有记忆冲突时用 update，过时的用 delete，不要重复添加
- 有时间的计划要写明日期（今天是 {datetime.now():%Y-%m-%d}）
- importance：5=核心身份/重大事件，3=一般喜好，1=琐碎"""
        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.3,
        )
        try:
            data = json.loads(resp.choices[0].message.content)
        except (json.JSONDecodeError, TypeError):
            return
        self.mem.apply_fact_ops(data.get("fact_ops") or [])
        ep = (data.get("episode") or "").strip()
        if ep:
            self.mem.add_episode(ep)
        self.mem.mark_processed(rows[-1]["id"])


_SENT_END = re.compile(r"[。！？!?~…\n]")
_STRIP = re.compile(r"（[^）]*）|\([^)]*\)|\*[^*]*\*|[\U0001F300-\U0001FAFF☀-➿]")


def clean_for_speech(text: str) -> str:
    return _STRIP.sub("", text).strip()


class SentenceSplitter:
    """把流式文本切成句子，方便逐句合成语音，降低延迟。"""

    def __init__(self):
        self.buf = ""

    def feed(self, delta: str) -> list[str]:
        self.buf += delta
        out = []
        while True:
            m = _SENT_END.search(self.buf)
            if not m:
                # 太长时在逗号处提前切
                if len(self.buf) > 40 and (i := max(self.buf.rfind("，"), self.buf.rfind(","))) > 8:
                    out.append(self.buf[:i + 1])
                    self.buf = self.buf[i + 1:]
                break
            end = m.end()
            while end < len(self.buf) and _SENT_END.match(self.buf[end]):
                end += 1
            out.append(self.buf[:end])
            self.buf = self.buf[end:]
        return [s for s in out if s.strip()]

    def flush(self) -> list[str]:
        s, self.buf = self.buf, ""
        return [s] if s.strip() else []
