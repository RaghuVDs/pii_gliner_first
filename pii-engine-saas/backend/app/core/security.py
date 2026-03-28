"""
Security utilities for the PII Engine SaaS platform.

Provides password hashing, JWT token creation/verification,
and API key generation/hashing.
"""

from __future__ import annotations

import hashlib
import logging
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple

from jose import JWTError, jwt
from passlib.context import CryptContext

from app.config import get_settings
from app.core.exceptions import UnauthorizedError

logger = logging.getLogger(__name__)

# ── Password Hashing ────────────────────────────────────────────────────

_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    """Hash a plain-text password using bcrypt.

    Args:
        password: The plain-text password.

    Returns:
        Bcrypt hash string.
    """
    return _pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain-text password against its bcrypt hash.

    Args:
        plain_password: The candidate password.
        hashed_password: The stored bcrypt hash.

    Returns:
        ``True`` if the password matches.
    """
    return _pwd_context.verify(plain_password, hashed_password)


# ── JWT Tokens ───────────────────────────────────────────────────────────


def create_access_token(
    data: Dict[str, Any],
    expires_delta: Optional[timedelta] = None,
) -> str:
    """Create a signed JWT access token.

    Args:
        data: Claims to encode in the token (must include ``sub``).
        expires_delta: Custom expiry duration; defaults to the
            configured ``ACCESS_TOKEN_EXPIRE_MINUTES``.

    Returns:
        Encoded JWT string.
    """
    settings = get_settings()

    to_encode = data.copy()
    now = datetime.now(timezone.utc)
    expire = now + (
        expires_delta
        if expires_delta is not None
        else timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire, "iat": now, "type": "access"})

    return jwt.encode(
        to_encode,
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM,
    )


def create_refresh_token(data: Dict[str, Any]) -> str:
    """Create a signed JWT refresh token with a longer expiry.

    Args:
        data: Claims to encode (must include ``sub``).

    Returns:
        Encoded JWT string.
    """
    settings = get_settings()

    to_encode = data.copy()
    now = datetime.now(timezone.utc)
    expire = now + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire, "iat": now, "type": "refresh"})

    return jwt.encode(
        to_encode,
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM,
    )


def decode_token(token: str) -> Dict[str, Any]:
    """Decode and validate a JWT token.

    Args:
        token: The encoded JWT string.

    Returns:
        The decoded payload dictionary.

    Raises:
        UnauthorizedError: If the token is invalid or expired.
    """
    settings = get_settings()
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
        )
        if payload.get("sub") is None:
            raise UnauthorizedError("Token missing 'sub' claim.")
        return payload
    except JWTError as exc:
        logger.warning("JWT decode failed: %s", exc)
        raise UnauthorizedError("Invalid or expired token.") from exc


# ── API Keys ─────────────────────────────────────────────────────────────

_API_KEY_PREFIX_LENGTH = 8
_API_KEY_SECRET_LENGTH = 40


def generate_api_key() -> Tuple[str, str, str]:
    """Generate a new API key with a human-readable prefix.

    Returns:
        A tuple of ``(key_string, key_hash, key_prefix)`` where:
        - ``key_string`` is the full key shown to the user once.
        - ``key_hash`` is the SHA-256 hash stored in the database.
        - ``key_prefix`` is the first 8 characters for display.
    """
    prefix = secrets.token_hex(_API_KEY_PREFIX_LENGTH // 2)  # 8 hex chars
    secret = secrets.token_urlsafe(_API_KEY_SECRET_LENGTH)
    key_string = f"pie_{prefix}_{secret}"
    key_hash = hash_api_key(key_string)
    return key_string, key_hash, f"pie_{prefix}"


def hash_api_key(key: str) -> str:
    """Produce a SHA-256 hash of an API key for secure storage.

    Args:
        key: The full API key string.

    Returns:
        Hex-encoded SHA-256 digest.
    """
    return hashlib.sha256(key.encode("utf-8")).hexdigest()
