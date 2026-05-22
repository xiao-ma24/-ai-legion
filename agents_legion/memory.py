import json
import uuid
from datetime import datetime
from pathlib import Path
from config import settings

DATA_DIR = Path(__file__).parent / "data"
SESSIONS_DIR = DATA_DIR / "sessions"
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)


class SessionMemory:
    def __init__(self, max_messages: int = 20):
        self._active: dict[str, list[dict]] = {}
        self.max_messages = max_messages

    def _path(self, session_id: str) -> Path:
        return SESSIONS_DIR / f"{session_id}.json"

    def create(self) -> str:
        session_id = str(uuid.uuid4())[:8]
        self._active[session_id] = []
        self._save(session_id)
        return session_id

    def add(self, session_id: str, role: str, content: str):
        msgs = self._load(session_id)
        msgs.append({"role": role, "content": content, "time": datetime.now().isoformat()})
        if len(msgs) > self.max_messages:
            msgs = msgs[-self.max_messages:]
        self._active[session_id] = msgs
        self._save(session_id)

    def get_history(self, session_id: str, last_n: int | None = None) -> list[dict]:
        msgs = self._load(session_id)
        if last_n:
            return msgs[-last_n:]
        return msgs

    def clear(self, session_id: str):
        self._active.pop(session_id, None)
        p = self._path(session_id)
        if p.exists():
            p.unlink()

    def list_sessions(self) -> list[dict]:
        sessions = []
        for p in sorted(SESSIONS_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
            sid = p.stem
            msgs = self._load(sid)
            title = "新对话"
            for m in msgs:
                if m["role"] == "user":
                    title = m["content"][:30]
                    break
            sessions.append({
                "id": sid,
                "title": title,
                "count": len(msgs),
                "updated": datetime.fromtimestamp(p.stat().st_mtime).strftime("%m-%d %H:%M"),
            })
        return sessions

    def _load(self, session_id: str) -> list[dict]:
        if session_id in self._active:
            return self._active[session_id]
        p = self._path(session_id)
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                self._active[session_id] = data
                return data
            except (json.JSONDecodeError, IOError):
                pass
        self._active[session_id] = []
        return []

    def _save(self, session_id: str):
        msgs = self._active.get(session_id, [])
        self._path(session_id).write_text(
            json.dumps(msgs, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def exists(self, session_id: str) -> bool:
        return self._path(session_id).exists()


memory = SessionMemory()
