import json
import logging
import time
import uuid
from typing import Callable
import redis
from rags.rag import RAG

class Chatbot:
    SESSIONS_ZSET = "chat:sessions"

    def __init__(self, rag: RAG, redis_client: redis.Redis):
        self.rag = rag
        self.redis = redis_client
        self.rag.cumulative_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def _hkey(self, sid: str) -> str: return f"chat:history:{sid}"
    def _mkey(self, sid: str) -> str: return f"chat:meta:{sid}"

    def create_session(self) -> str:
        sid = str(uuid.uuid4())
        now = time.time()
        meta = {"session_id": sid, "title": "New Chat", "created_at": now, "updated_at": now}
        self.redis.set(self._mkey(sid), json.dumps(meta))
        self.redis.zadd(self.SESSIONS_ZSET, {sid: now})
        return sid

    def list_sessions(self) -> list[dict]:
        sids = self.redis.zrevrange(self.SESSIONS_ZSET, 0, -1)
        result = []
        for sid in sids:
            raw = self.redis.get(self._mkey(sid))
            if raw:
                result.append(json.loads(raw))
        return result

    def delete_session(self, session_id: str) -> None:
        self.redis.delete(self._hkey(session_id))
        self.redis.delete(self._mkey(session_id))
        self.redis.zrem(self.SESSIONS_ZSET, session_id)

    def get_display_history(self, session_id: str) -> list[dict]:
        history = self._load_history(session_id)
        visible = []
        for turn in history:
            if turn["role"] == "user":
                visible.append({"role": "user", "content": turn.get("display", turn["content"])})
            else:
                visible.append({"role": "assistant", "content": turn["content"]})
        return visible

    def _load_history(self, session_id: str) -> list[dict]:
        raw = self.redis.get(self._hkey(session_id))
        return json.loads(raw) if raw else []

    def _save_history(self, session_id: str, history: list[dict]) -> None:
        self.redis.set(self._hkey(session_id), json.dumps(history))

    def _touch_meta(self, session_id: str, title: str | None = None) -> None:
        now = time.time()
        raw = self.redis.get(self._mkey(session_id))
        meta = json.loads(raw) if raw else {
            "session_id": session_id, "title": "New Chat", "created_at": now
        }
        meta["updated_at"] = now
        if title and meta["title"] == "New Chat":
            meta["title"] = title
        self.redis.set(self._mkey(session_id), json.dumps(meta))
        self.redis.zadd(self.SESSIONS_ZSET, {session_id: now})

    def chat(
        self,
        query: str,
        session_id: str,
        confirm_external: Callable[[], bool] | None = None,
    ):
        history = self._load_history(session_id)
        title = query[:60] + ("…" if len(query) > 60 else "")
        self._touch_meta(session_id, title)

        query_embedding = self.rag.embed_query(query)
        chunks = self.rag.retrieve(query_embedding)

        if not chunks:
            rephrased = self.rag.rephrase_query(query, history=history)
            chunks = self.rag.retrieve(self.rag.embed_query(rephrased))

        context = self.rag.build_context(chunks) if chunks else ""

        if not context:
            if confirm_external is not None and not confirm_external():
                return None

        llm_content = f"QUESTION: '{query}'\n\n{context}\nANSWER:" if context else query
        history.append({"role": "user", "content": llm_content, "display": query})
        self._save_history(session_id, history)

        messages = [
            {"role": "system", "content": self.rag.config.ANSWER_PROMPT},
            *[{"role": t["role"], "content": t["content"]} for t in history],
        ]

        response = self.rag.client.chat.completions.create(
            model=self.rag.llm_model,
            messages=messages
        )

        answer = response.choices[0].message.content
        history.append({"role": "assistant", "content": answer})
        self._save_history(session_id, history)
        self._touch_meta(session_id)
        self._track_tokens(response.usage)
        self._log(query, context, answer)

        return answer, response.usage, self.rag.cumulative_tokens

    def _track_tokens(self, usage) -> None:
        if self.rag.client.provider != "openai" or not usage:
            return
        self.rag.cumulative_tokens["prompt_tokens"]     += usage.prompt_tokens
        self.rag.cumulative_tokens["completion_tokens"] += usage.completion_tokens
        self.rag.cumulative_tokens["total_tokens"]      += usage.total_tokens

    def _log(self, query: str, context: str, answer: str):
        logging.info("[QUESTION]: %s", query)
        logging.info("[CONTEXT]: %s", context)
        logging.info("[ANSWER]: %s", answer)