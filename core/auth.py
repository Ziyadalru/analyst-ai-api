from fastapi import Header, HTTPException
from jose import jwt, JWTError
from core.config import settings


async def get_current_user(authorization: str = Header(default=None)) -> dict:
    """Decode Supabase JWT and return {id, email}. Raises 401 if invalid."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
    token = authorization.split(" ", 1)[1]
    if not settings.SUPABASE_JWT_SECRET:
        raise HTTPException(status_code=500, detail="JWT secret not configured")
    try:
        payload = jwt.decode(
            token,
            settings.SUPABASE_JWT_SECRET,
            algorithms=["HS256"],
            audience="authenticated",
        )
        return {"id": payload["sub"], "email": payload.get("email", "")}
    except JWTError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {e}")


async def get_optional_user(authorization: str = Header(default=None)) -> dict | None:
    """Like get_current_user but returns None instead of raising for unauthenticated requests."""
    if not authorization or not authorization.startswith("Bearer "):
        return None
    try:
        return await get_current_user(authorization)
    except HTTPException:
        return None
