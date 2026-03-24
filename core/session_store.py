"""
Session store: Redis-backed with in-process hot cache.
Falls back to /tmp pickle if Redis is unavailable (local dev).
Sessions auto-expire after 24 hours (TTL set on Redis keys).
"""
import uuid
import os
import time
import pickle
import io
import pandas as pd
from typing import Optional

_TTL = 24 * 60 * 60  # 24 hours

# ── Redis connection (optional) ────────────────────────────────────────────────
_redis = None

def _get_redis():
    global _redis
    if _redis is not None:
        return _redis
    redis_url = os.environ.get("REDIS_URL") or os.environ.get("REDIS_PRIVATE_URL")
    if not redis_url:
        return None
    try:
        import redis
        _redis = redis.from_url(redis_url, decode_responses=False, socket_timeout=3)
        _redis.ping()
        return _redis
    except Exception:
        _redis = None
        return None


# ── Local fallback (pickle on disk) ───────────────────────────────────────────
_CACHE_DIR = "/tmp/analyst_ai_sessions"
_store: dict[str, tuple[pd.DataFrame, dict]] = {}  # hot cache


def _disk_path(session_id: str) -> str:
    os.makedirs(_CACHE_DIR, exist_ok=True)
    return os.path.join(_CACHE_DIR, f"{session_id}.pkl")


def _cleanup_disk_expired() -> None:
    now = time.time()
    try:
        for fname in os.listdir(_CACHE_DIR):
            fpath = os.path.join(_CACHE_DIR, fname)
            if os.path.isfile(fpath) and now - os.path.getmtime(fpath) > _TTL:
                os.remove(fpath)
                _store.pop(fname.replace(".pkl", ""), None)
    except Exception:
        pass


# ── Serialisation helpers ──────────────────────────────────────────────────────
def _serialize(df: pd.DataFrame, cols: dict) -> bytes:
    buf = io.BytesIO()
    pickle.dump((df, cols), buf)
    return buf.getvalue()


def _deserialize(data: bytes) -> tuple[pd.DataFrame, dict]:
    return pickle.loads(data)


# ── Public API ─────────────────────────────────────────────────────────────────
def save_session(df: pd.DataFrame, cols: dict) -> str:
    session_id = str(uuid.uuid4())
    _store[session_id] = (df, cols)

    r = _get_redis()
    if r:
        try:
            r.setex(f"session:{session_id}", _TTL, _serialize(df, cols))
            return session_id
        except Exception:
            pass

    # Fallback: disk
    _cleanup_disk_expired()
    with open(_disk_path(session_id), "wb") as f:
        pickle.dump((df, cols), f)
    return session_id


def save_df(df: pd.DataFrame) -> str:
    return save_session(df, {})


def _load(session_id: str) -> Optional[tuple[pd.DataFrame, dict]]:
    if session_id in _store:
        return _store[session_id]

    r = _get_redis()
    if r:
        try:
            data = r.get(f"session:{session_id}")
            if data:
                entry = _deserialize(data)
                _store[session_id] = entry
                return entry
        except Exception:
            pass

    # Fallback: disk
    p = _disk_path(session_id)
    if os.path.exists(p):
        with open(p, "rb") as f:
            entry = pickle.load(f)
        _store[session_id] = entry
        return entry

    return None


def get_df(session_id: str) -> Optional[pd.DataFrame]:
    entry = _load(session_id)
    return entry[0] if entry else None


def get_cols(session_id: str) -> Optional[dict]:
    entry = _load(session_id)
    return entry[1] if entry else None


def delete_df(session_id: str) -> None:
    _store.pop(session_id, None)
    r = _get_redis()
    if r:
        try:
            r.delete(f"session:{session_id}")
        except Exception:
            pass
    p = _disk_path(session_id)
    if os.path.exists(p):
        os.remove(p)
