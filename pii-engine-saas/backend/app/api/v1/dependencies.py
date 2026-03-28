"""
FastAPI dependency functions for authentication, tenant resolution,
and role-based access control.

These are injected into route handlers via ``Depends(...)`` and provide
a clean separation between HTTP-layer concerns and business logic.
"""

from __future__ import annotations

import uuid
from typing import Callable, List

from fastapi import Depends, Header, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.exceptions import ForbiddenError, UnauthorizedError
from app.models.tenant import Tenant
from app.models.user import User

# ---------------------------------------------------------------------------
# Security scheme -- extracts the Bearer token from the Authorization header
# ---------------------------------------------------------------------------
_bearer_scheme = HTTPBearer(auto_error=False)


# ---------------------------------------------------------------------------
# JWT token decoding (placeholder -- will be implemented in core/security.py)
# ---------------------------------------------------------------------------
async def _decode_access_token(token: str) -> dict:
    """Decode and validate a JWT access token.

    Returns the token payload containing at minimum:
        - sub (user_id)
        - tenant_id
        - role
    """
    from app.core.security import decode_token  # noqa: E402

    return decode_token(token)


# ---------------------------------------------------------------------------
# get_current_user -- resolve the authenticated user from the JWT
# ---------------------------------------------------------------------------
async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    """Extract and validate the JWT from the ``Authorization`` header,
    then look up the corresponding user record.

    Raises:
        UnauthorizedError: If the token is missing, expired, or invalid,
            or if the user no longer exists / is deactivated.
    """
    if credentials is None:
        raise UnauthorizedError("Missing authentication token.")

    payload = await _decode_access_token(credentials.credentials)
    user_id: str | None = payload.get("sub")

    if user_id is None:
        raise UnauthorizedError("Invalid token payload.")

    result = await db.execute(
        select(User).where(
            User.id == uuid.UUID(user_id),
            User.deleted_at.is_(None),
        )
    )
    user = result.scalar_one_or_none()

    if user is None:
        raise UnauthorizedError("User not found or deactivated.")

    if not user.is_active:
        raise UnauthorizedError("User account is deactivated.")

    return user


# ---------------------------------------------------------------------------
# get_current_tenant -- resolve the tenant from the authenticated user
# ---------------------------------------------------------------------------
async def get_current_tenant(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Tenant:
    """Resolve the tenant record for the currently authenticated user.

    Raises:
        UnauthorizedError: If the tenant is not found or deactivated.
    """
    result = await db.execute(
        select(Tenant).where(
            Tenant.id == current_user.tenant_id,
            Tenant.deleted_at.is_(None),
        )
    )
    tenant = result.scalar_one_or_none()

    if tenant is None:
        raise UnauthorizedError("Tenant not found or deactivated.")

    if not tenant.is_active:
        raise ForbiddenError("Tenant account is suspended.")

    return tenant


# ---------------------------------------------------------------------------
# require_role -- role-based access control dependency factory
# ---------------------------------------------------------------------------

# Role hierarchy: admin > analyst > viewer
_ROLE_HIERARCHY: dict[str, int] = {
    "viewer": 0,
    "analyst": 1,
    "admin": 2,
}


def require_role(*allowed_roles: str) -> Callable:
    """Return a FastAPI dependency that enforces role-based access.

    Usage::

        @router.get("/", dependencies=[Depends(require_role("admin"))])
        async def admin_only_route(...): ...

        @router.get("/", dependencies=[Depends(require_role("admin", "analyst"))])
        async def analyst_and_above(...): ...

    When a single role is passed, all roles at that level **or above** in the
    hierarchy are granted access.  When multiple roles are passed, access is
    granted if the user holds **any** of the listed roles.
    """

    async def _check_role(
        current_user: User = Depends(get_current_user),
    ) -> User:
        user_level = _ROLE_HIERARCHY.get(current_user.role, -1)

        if len(allowed_roles) == 1:
            # Hierarchical check: user level must be >= required level
            required_level = _ROLE_HIERARCHY.get(allowed_roles[0], 99)
            if user_level < required_level:
                raise ForbiddenError(
                    f"This action requires the '{allowed_roles[0]}' role or above."
                )
        else:
            # Explicit list check
            if current_user.role not in allowed_roles:
                raise ForbiddenError(
                    f"This action requires one of the following roles: "
                    f"{', '.join(allowed_roles)}."
                )

        return current_user

    return _check_role
