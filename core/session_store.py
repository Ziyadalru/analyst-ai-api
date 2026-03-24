"""
Session store: in-memory cache backed by disk (pickle) so sessions survive restarts.
Keyed by session_id (UUID). For production, replace with Redis + Parquet.
"""
import uuid
import os
import pickle
import pandas as pd
from typing import Optional

_CACHE_DIR = "/tmp/analyst_ai_sessions"
os.makedirs(_CACHE_DIR, exist_ok=True)

# Hot cache — avoids disk reads on repeated access
_store: dict[str, tuple[pd.DataFrame, dict]] = {}


def _path(session_id: str) -> str:
    return os.path.join(_CACHE_DIR, f"{session_id}.pkl")


def save_session(df: pd.DataFrame, cols: dict) -> str:
    session_id = str(uuid.uuid4())
    _store[session_id] = (df, cols)
    with open(_path(session_id), "wb") as f:
        pickle.dump((df, cols), f)
    return session_id


def save_df(df: pd.DataFrame) -> str:
    return save_session(df, {})


def get_df(session_id: str) -> Optional[pd.DataFrame]:
    if session_id in _store:
        return _store[session_id][0]
    # Try loading from disk after a restart
    p = _path(session_id)
    if os.path.exists(p):
        with open(p, "rb") as f:
            entry = pickle.load(f)
        _store[session_id] = entry
        return entry[0]
    return None


def get_cols(session_id: str) -> Optional[dict]:
    if session_id in _store:
        return _store[session_id][1]
    p = _path(session_id)
    if os.path.exists(p):
        with open(p, "rb") as f:
            entry = pickle.load(f)
        _store[session_id] = entry
        return entry[1]
    return None


def delete_df(session_id: str) -> None:
    _store.pop(session_id, None)
    p = _path(session_id)
    if os.path.exists(p):
        os.remove(p)
