from fastapi import APIRouter, Depends
from core.auth import get_current_user
from core.supabase_client import get_usage_today
from core.config import settings

router = APIRouter()


@router.get("/usage")
async def get_usage(user: dict = Depends(get_current_user)):
    used = get_usage_today(user["id"])
    limit = settings.FREE_LIMIT
    return {
        "used": used,
        "limit": limit,
        "remaining": max(0, limit - used),
    }
