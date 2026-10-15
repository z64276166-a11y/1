"""长期记忆：SQLite 存储聊天记录、关于用户的事实、过往事件摘要。"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

DB_PATH = Path(__file__).parent / "data" / "memory.db"


def _bigrams(text: str) -> set:
    t = "".join(ch for ch in text.lower() if ch.isalnum())
    return {t[i:i + 2] for i in range(len(t) - 1)}


class Memory:
    def __init__(self, path: Path = DB_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS messages(
                id INTEGER PRIMARY KEY, role TEXT, content TEXT, ts REAL,
                processed INTEGER DEFAULT 0);
            CREATE TABLE IF NOT EXISTS facts(
                id INTEGER PRIMARY KEY, content TEXT, importance INTEGER,
                created REAL, updated REAL);
            CREATE TABLE IF NOT EXISTS episodes(
                id INTEGER PRIMARY KEY, summary TEXT, ts REAL);
            """
        )
        self.db.commit()

    # ---------- 聊天记录 ----------
    def add_message(self, role: str, content: str):
        self.db.execute("INSERT INTO messages(role, content, ts) VALUES(?,?,?)",
                        (role, content, time.time()))
        self.db.commit()

    def recent_messages(self, n: int = 24) -> list[dict]:
        rows = self.db.execute(
            "SELECT role, content FROM messages ORDER BY id DESC LIMIT ?", (n,)).fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

    def last_chat_time(self) -> float | None:
        row = self.db.execute("SELECT MAX(ts) AS t FROM messages").fetchone()
        return row["t"]

    def unprocessed(self) -> list[sqlite3.Row]:
        return self.db.execute(
            "SELECT id, role, content, ts FROM messages WHERE processed=0 ORDER BY id").fetchall()

    def mark_processed(self, max_id: int):
        self.db.execute("UPDATE messages SET processed=1 WHERE id<=?", (max_id,))
        self.db.commit()

    # ---------- 事实 ----------
    def facts(self, limit: int = 80) -> list[sqlite3.Row]:
        return self.db.execute(
            "SELECT id, content, importance FROM facts "
            "ORDER BY importance DESC, updated DESC LIMIT ?", (limit,)).fetchall()

    def apply_fact_ops(self, ops: list[dict]):
        now = time.time()
        for op in ops:
            kind = op.get("op", "add")
            content = (op.get("content") or "").strip()
            imp = int(op.get("importance", 3) or 3)
            fid = op.get("id")
            if kind == "add" and content:
                self.db.execute(
                    "INSERT INTO facts(content, importance, created, updated) VALUES(?,?,?,?)",
                    (content, imp, now, now))
            elif kind == "update" and fid and content:
                self.db.execute("UPDATE facts SET content=?, importance=?, updated=? WHERE id=?",
                                (content, imp, now, fid))
            elif kind == "delete" and fid:
                self.db.execute("DELETE FROM facts WHERE id=?", (fid,))
        self.db.commit()

    # ---------- 事件摘要 ----------
    def add_episode(self, summary: str):
        self.db.execute("INSERT INTO episodes(summary, ts) VALUES(?,?)", (summary, time.time()))
        self.db.commit()

    def relevant_episodes(self, query: str, k_recent: int = 4, k_match: int = 4) -> list[sqlite3.Row]:
        rows = self.db.execute("SELECT id, summary, ts FROM episodes ORDER BY id DESC").fetchall()
        recent = rows[:k_recent]
        q = _bigrams(query)
        scored = []
        for r in rows[k_recent:]:
            s = len(q & _bigrams(r["summary"]))
            if s >= 2:
                scored.append((s, r))
        scored.sort(key=lambda x: -x[0])
        picked = recent + [r for _, r in scored[:k_match]]
        return sorted(picked, key=lambda r: r["ts"])

    def export_json(self) -> str:
        return json.dumps({
            "facts": [dict(r) for r in self.facts(1000)],
            "episodes": [dict(r) for r in self.db.execute("SELECT * FROM episodes").fetchall()],
        }, ensure_ascii=False, indent=2)
