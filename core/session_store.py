"""
In-memory session store for uploaded DataFrames + detected columns.
Keyed by session_id (UUID). For production, replace with Redis + Parquet.
"""
import uuid
import pandas as pd
from typing import Optional

# Stores (DataFrame, cols_dict) tuples
_store: dict[str, tuple[pd.DataFrame, dict]] = {}


def save_session(df: pd.DataFrame, cols: dict) -> str:
    session_id = str(uuid.uuid4())
    _store[session_id] = (df, cols)
    return session_id


def save_df(df: pd.DataFrame) -> str:
    """Backward-compatible: save without cols."""
    return save_session(df, {})


def get_df(session_id: str) -> Optional[pd.DataFrame]:
    entry = _store.get(session_id)
    return entry[0] if entry else None


def get_cols(session_id: str) -> Optional[dict]:
    entry = _store.get(session_id)
    return entry[1] if entry else None


def delete_df(session_id: str) -> None:
    _store.pop(session_id, None)
