"""
Tenant context middleware.

Extracts the tenant identifier from the authenticated request
(JWT claims, API key, or explicit header) and makes it available
on ``request.state.tenant_id`` for all downstream handlers.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

logger = logging.getLogger(__name__)

# Header that can be used by super-admins to impersonate a tenant
TENANT_HEADER = "X-Tenant-Id"


class TenantContextMiddleware(BaseHTTPMiddleware):
    """Populate ``request.state.tenant_id`` from multiple sources.

    Resolution order:
    1. ``X-Tenant-Id`` header (only honoured for super-admin users).
    2. ``tenant_id`` claim from the decoded JWT (set by AuthMiddleware).
    3. ``tenant_id`` from API key lookup (set by AuthMiddleware).

    If none of these sources provide a value, ``tenant_id`` remains
    ``None`` and route-level dependencies decide whether to reject
    the request.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        tenant_id: Optional[str] = None

        # 1. Explicit header override (super-admin impersonation)
        header_tenant = request.headers.get(TENANT_HEADER)
        user_role: Optional[int] = getattr(request.state, "user_role", None)

        if header_tenant and user_role is not None:
            # Role value 50 == SUPER_ADMIN in our Role enum
            if user_role >= 50:
                tenant_id = header_tenant
                logger.info(
                    "Super-admin tenant override: tenant_id=%s user=%s",
                    tenant_id,
                    getattr(request.state, "user_id", "unknown"),
                )
            else:
                logger.warning(
                    "Non-super-admin attempted tenant override: user=%s header=%s",
                    getattr(request.state, "user_id", "unknown"),
                    header_tenant,
                )

        # 2. Fall back to JWT-derived tenant_id (set by AuthMiddleware)
        if tenant_id is None:
            tenant_id = getattr(request.state, "tenant_id", None)

        # Set the final resolved tenant_id
        request.state.tenant_id = tenant_id

        response = await call_next(request)
        return response
