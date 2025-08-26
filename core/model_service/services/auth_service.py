from sqlalchemy.orm import Session
from sqlalchemy import select, update
from datetime import timedelta
from uuid import uuid4
import hashlib

from ..models import User, RefreshToken
from ..security import (
    verify_password, hash_password,
    create_access_token, create_refresh_token, decode_token, now_utc
)
from ..config import get_settings

settings = get_settings()

def _hash_refresh_token(raw_token: str) -> str:
    # Never store raw refresh tokens; hash them (constant-time compare on read)
    return hashlib.sha256(raw_token.encode("utf-8")).hexdigest()

class AuthService:
    # --------- USER AUTH ----------
    def authenticate_user(self, db: Session, email: str, password: str) -> User | None:
        user = db.scalar(select(User).where(User.email == email))
        if not user or not verify_password(password, user.password_hash):
            return None
        if not user.is_active:
            return None
        return user

    def signup_user(self, db: Session, email: str, password: str, is_creator: bool = False) -> User:
        user = User(email=email, password_hash=hash_password(password), is_creator=is_creator)
        db.add(user)
        db.commit()
        db.refresh(user)
        return user

    # --------- TOKEN PAIR ----------
    def issue_token_pair(self, db: Session, user: User) -> tuple[str, str]:
        # jti for refresh rotation
        jti = str(uuid4())
        access = create_access_token({"sub": user.email, "uid": user.id})
        refresh = create_refresh_token({"sub": user.email, "uid": user.id, "jti": jti})

        token_hash = _hash_refresh_token(refresh)
        expires_at = now_utc() + timedelta(days=settings.REFRESH_TOKEN_TTL_DAYS)

        db.add(RefreshToken(jti=jti, user_id=user.id, token_hash=token_hash, expires_at=expires_at))
        db.commit()
        return access, refresh

    def rotate_refresh(self, db: Session, raw_refresh_token: str) -> tuple[str, str] | None:
        payload = decode_token(raw_refresh_token)
        if not payload or payload.get("scope") != "refresh":
            return None

        jti = payload.get("jti")
        uid = payload.get("uid")
        if not jti or not uid:
            return None

        # Validate stored token
        stored: RefreshToken | None = db.scalar(select(RefreshToken).where(RefreshToken.jti == jti))
        if not stored or stored.revoked:
            return None
        if stored.expires_at < now_utc():
            return None
        if stored.token_hash != _hash_refresh_token(raw_refresh_token):
            return None  # token mismatch

        # Revoke current and issue new
        stored.revoked = True
        new_jti = str(uuid4())
        new_access = create_access_token({"sub": payload["sub"], "uid": uid})
        new_refresh = create_refresh_token({"sub": payload["sub"], "uid": uid, "jti": new_jti})

        db.add(RefreshToken(
            jti=new_jti,
            user_id=uid,
            token_hash=_hash_refresh_token(new_refresh),
            expires_at=now_utc() + (stored.expires_at - stored.created_at)
        ))
        db.commit()
        return new_access, new_refresh

    def revoke_all_user_refresh_tokens(self, db: Session, user_id: int) -> int:
        res = db.execute(
            update(RefreshToken)
            .where(RefreshToken.user_id == user_id, RefreshToken.revoked == False)
            .values(revoked=True)
        )
        db.commit()
        return res.rowcount
