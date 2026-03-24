from fastapi import Depends, HTTPException
from core.auth import get_optional_user
from core.supabase_client import get_usage_today, increment_usage
from core.config import settings


async def check_usage_gate(user: dict | None = Depends(get_optional_user)) -> dict | None:
    """
    Authenticated users: check Supabase usage table, raise 429 if over limit.
    Anonymous users: pass through (frontend tracks anon_count in localStorage).
    """
    if user is None:
        return None
    used = get_usage_today(user["id"])
    if used >= settings.FREE_LIMIT:
        raise HTTPException(
            status_code=429,
            detail={
                "error": "daily_limit_reached",
                "used": used,
                "limit": settings.FREE_LIMIT,
            },
        )
    return user
