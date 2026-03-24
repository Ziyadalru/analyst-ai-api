import os
from datetime import date
from supabase import create_client
from dotenv import load_dotenv

# Try multiple locations for .env
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))
load_dotenv(os.path.join(os.getcwd(), '.env'))  # fallback: cwd (project root when streamlit runs)

# Read from env (already loaded above)
_url  = os.getenv("SUPABASE_URL", "")
_anon = os.getenv("SUPABASE_ANON_KEY", "")
_svc  = os.getenv("SUPABASE_SERVICE_KEY", "")

# Override with Streamlit secrets if available (for deployment)
try:
    import streamlit as st
    if hasattr(st, "secrets"):
        _url  = st.secrets.get("SUPABASE_URL", _url)
        _anon = st.secrets.get("SUPABASE_ANON_KEY", _anon)
        _svc  = st.secrets.get("SUPABASE_SERVICE_KEY", _svc)
except Exception:
    pass

FREE_LIMIT = 10

_db_key = _svc if (_svc and not _svc.startswith("your_")) else _anon

try:
    _auth_client = create_client(_url, _anon) if (_url and _anon) else None
except Exception as _e:
    _auth_client = None
    _INIT_ERROR = str(_e)
else:
    _INIT_ERROR = ""

try:
    _db_client = create_client(_url, _db_key) if (_url and _db_key) else None
except Exception:
    _db_client = _auth_client  # fallback to same client


def sign_up(email: str, password: str):
    """Returns (user, error_message)."""
    if not _auth_client:
        return None, f"Auth service error: URL='{_url[:40]}' ERR='{_INIT_ERROR}'"
    try:
        res = _auth_client.auth.sign_up({"email": email, "password": password})
        if res.user:
            return res.user, None
        return None, "Sign-up failed. Try a different email."
    except Exception as e:
        return None, str(e)


def sign_in(email: str, password: str):
    """Returns (user, session, error_message)."""
    if not _auth_client:
        return None, None, f"Auth service error: URL='{_url[:40]}' ERR='{_INIT_ERROR}'"
    try:
        res = _auth_client.auth.sign_in_with_password({"email": email, "password": password})
        if res.user and res.session:
            return res.user, res.session, None
        return None, None, "Invalid email or password."
    except Exception as e:
        return None, None, str(e)


def sign_out():
    if not _auth_client:
        return
    try:
        _auth_client.auth.sign_out()
    except Exception:
        pass


def get_usage_today(user_id: str) -> int:
    if not _db_client:
        return 0
    try:
        res = (
            _db_client.table("usage")
            .select("count")
            .eq("user_id", user_id)
            .eq("date", str(date.today()))
            .execute()
        )
        return res.data[0]["count"] if res.data else 0
    except Exception:
        return 0


def increment_usage(user_id: str) -> int:
    if not _db_client:
        return 0
    try:
        today = str(date.today())
        res = _db_client.table("usage").select("count").eq("user_id", user_id).eq("date", today).execute()
        if res.data:
            new_count = res.data[0]["count"] + 1
            _db_client.table("usage").update({"count": new_count}).eq("user_id", user_id).eq("date", today).execute()
        else:
            new_count = 1
            _db_client.table("usage").insert({"user_id": user_id, "date": today, "count": 1}).execute()
        return new_count
    except Exception:
        return 0


def can_run_analysis(user_id: str) -> tuple:
    """Returns (allowed: bool, used: int, limit: int)."""
    used = get_usage_today(user_id)
    return used < FREE_LIMIT, used, FREE_LIMIT
