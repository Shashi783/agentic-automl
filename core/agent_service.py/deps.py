from fastapi import Depends, HTTPException, status
from auth import verify_dummy_token

async def get_current_user(authorization: str = "Bearer student-token"):
    """
    Dummy get_current_user dependency.
    Always accepts 'student-token' or 'creator-token' as Bearer tokens.
    """
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Invalid auth scheme")

    user = verify_dummy_token(token)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )
    return user
