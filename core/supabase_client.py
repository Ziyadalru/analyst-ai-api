from datetime import date
from supabase import create_client
from core.config import settings

_db_client = None

def get_db_client():
    global _db_client
    if _db_client is None:
        key = settings.SUPABASE_SERVICE_KEY or settings.SUPABASE_ANON_KEY
        if settings.SUPABASE_URL and key:
            _db_client = create_client(settings.SUPABASE_URL, key)
    return _db_client


def get_usage_today(user_id: str) -> int:
    client = get_db_client()
    if not client:
        return 0
    try:
        res = (
            client.table("usage")
            .select("count")
            .eq("user_id", user_id)
            .eq("date", str(date.today()))
            .execute()
        )
        return res.data[0]["count"] if res.data else 0
    except Exception:
        return 0


def increment_usage(user_id: str) -> int:
    client = get_db_client()
    if not client:
        return 0
    try:
        today = str(date.today())
        res = client.table("usage").select("count").eq("user_id", user_id).eq("date", today).execute()
        if res.data:
            new_count = res.data[0]["count"] + 1
            client.table("usage").update({"count": new_count}).eq("user_id", user_id).eq("date", today).execute()
        else:
            new_count = 1
            client.table("usage").insert({"user_id": user_id, "date": today, "count": 1}).execute()
        return new_count
    except Exception:
        return 0
