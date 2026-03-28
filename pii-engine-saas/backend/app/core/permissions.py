"""
Role-based access control (RBAC) for the PII Engine SaaS platform.

Defines roles, their permitted actions, and FastAPI dependencies
for enforcing role and scope requirements.
"""

from __future__ import annotations

import logging
from enum import IntEnum
from typing import TYPE_CHECKING, Callable, List, Set

from fastapi import Depends, Request

from app.core.exceptions import ForbiddenError, UnauthorizedError

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ── Roles ────────────────────────────────────────────────────────────────


class Role(IntEnum):
    """
    User roles ordered by privilege level (higher = more privileged).

    The integer value encodes hierarchy so that a simple ``>=``
    comparison can enforce minimum-role checks.
    """

    VIEWER = 10
    API_CONSUMER = 20
    ANALYST = 30
    ADMIN = 40
    SUPER_ADMIN = 50


# ── Permissions Map ──────────────────────────────────────────────────────

# Each role inherits the permissions of all lower roles implicitly
# because the ``require_role`` dependency uses ``>=`` comparison.
# This map lists the *additional* named actions granted at each level.

ROLE_PERMISSIONS: dict[Role, Set[str]] = {
    Role.VIEWER: {
        "detections:read",
        "reports:read",
        "dashboard:read",
    },
    Role.API_CONSUMER: {
        "detections:read",
        "detections:create",
        "api_keys:read_own",
    },
    Role.ANALYST: {
        "detections:read",
        "detections:create",
        "detections:export",
        "reports:read",
        "reports:create",
        "dashboard:read",
        "rules:read",
        "rules:propose",
        "training:read",
        "training:create",
        "jobs:read",
        "jobs:create",
    },
    Role.ADMIN: {
        "detections:read",
        "detections:create",
        "detections:export",
        "detections:delete",
        "reports:read",
        "reports:create",
        "reports:delete",
        "dashboard:read",
        "rules:read",
        "rules:propose",
        "rules:approve",
        "rules:delete",
        "training:read",
        "training:create",
        "training:delete",
        "jobs:read",
        "jobs:create",
        "jobs:cancel",
        "users:read",
        "users:create",
        "users:update",
        "users:deactivate",
        "api_keys:read",
        "api_keys:create",
        "api_keys:revoke",
        "tenant:read",
        "tenant:update",
        "audit:read",
    },
    Role.SUPER_ADMIN: {
        "*",  # wildcard: all actions
    },
}


def _get_all_permissions_for_role(role: Role) -> Set[str]:
    """Collect the full set of permissions for a role, including inherited ones."""
    perms: Set[str] = set()
    for r in Role:
        if r <= role:
            perms |= ROLE_PERMISSIONS.get(r, set())
    return perms


def role_has_permission(role: Role, action: str) -> bool:
    """Check whether a role is allowed to perform a named action.

    Args:
        role: The user's role.
        action: The action string (e.g. ``detections:create``).

    Returns:
        ``True`` if permitted.
    """
    perms = _get_all_permissions_for_role(role)
    return "*" in perms or action in perms


# ── FastAPI Dependencies ─────────────────────────────────────────────────


def require_role(minimum_role: Role) -> Callable:
    """Return a FastAPI dependency that enforces a minimum role level.

    Usage::

        @router.get("/admin-only", dependencies=[Depends(require_role(Role.ADMIN))])
        async def admin_endpoint(): ...

    Args:
        minimum_role: The lowest role allowed to access the endpoint.
    """

    async def _check_role(request: Request) -> None:
        user = getattr(request.state, "current_user", None)
        if user is None:
            raise UnauthorizedError("Authentication required.")

        user_role_value: int | None = getattr(user, "role", None)
        if user_role_value is None:
            raise ForbiddenError("User has no assigned role.")

        try:
            user_role = Role(user_role_value)
        except ValueError:
            raise ForbiddenError(f"Unknown role value: {user_role_value}")

        if user_role < minimum_role:
            logger.warning(
                "Access denied: user=%s role=%s required=%s",
                getattr(user, "id", "unknown"),
                user_role.name,
                minimum_role.name,
            )
            raise ForbiddenError(
                f"Requires at least {minimum_role.name} role."
            )

    return _check_role


def require_scope(scope: str) -> Callable:
    """Return a FastAPI dependency that checks an API key scope.

    API keys carry a list of granted scopes. This dependency ensures
    the request's API key includes the required scope.

    Args:
        scope: The required scope string (e.g. ``detections:create``).
    """

    async def _check_scope(request: Request) -> None:
        api_key_scopes: List[str] | None = getattr(
            request.state, "api_key_scopes", None
        )
        if api_key_scopes is None:
            raise UnauthorizedError("API key authentication required.")

        if "*" not in api_key_scopes and scope not in api_key_scopes:
            logger.warning(
                "Scope denied: required=%s available=%s", scope, api_key_scopes
            )
            raise ForbiddenError(f"API key missing required scope: {scope}")

    return _check_scope
