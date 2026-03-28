"""
Masking rule management endpoints.

Controls which masking strategy (e.g. redact, hash, mask, encrypt,
tokenise) is applied to each PII type.  System defaults exist for every
PII type; tenants can override the strategy per type.
"""

from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, Query, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.api.v1.dependencies import (
    get_current_tenant,
    get_current_user,
    require_role,
)
from app.models.tenant import Tenant
from app.models.user import User
from app.schemas.common import ErrorResponse, SuccessResponse
from app.services import masking_rule_service

router = APIRouter(tags=["Masking Rules"])


# ---------------------------------------------------------------------------
# Inline request / response schemas
# ---------------------------------------------------------------------------

class MaskingStrategyInfo(BaseModel):
    """Description of a single available masking strategy."""
    name: str
    display_name: str
    description: str
    example_input: str
    example_output: str
    supports_reversible: bool = False


class MaskingRuleResponse(BaseModel):
    pii_type_name: str
    system_strategy: str = Field(..., description="System-default strategy")
    tenant_strategy: str | None = Field(None, description="Tenant override (null = use system)")
    effective_strategy: str = Field(..., description="Strategy that will actually be used")
    strategy_config: dict | None = Field(None, description="Additional strategy parameters")

    model_config = ConfigDict(from_attributes=True)


class MaskingRuleUpdate(BaseModel):
    strategy: str = Field(
        ...,
        description="Masking strategy: redact, hash, mask, encrypt, tokenise, custom",
    )
    strategy_config: dict | None = Field(
        None,
        description="Optional strategy parameters (e.g. mask_char, hash_algorithm)",
    )


# ---------------------------------------------------------------------------
# GET / -- list all masking rules + tenant overrides
# ---------------------------------------------------------------------------
@router.get(
    "/",
    status_code=status.HTTP_200_OK,

    summary="List all masking rules",
)
async def list_masking_rules(
    db: AsyncSession = Depends(get_db),
    tenant: Tenant = Depends(get_current_tenant),
    pii_type_name: str | None = Query(None, description="Filter by PII type name"),
):
    """Return all masking rules with system defaults and tenant overrides
    merged to show the effective strategy for each PII type."""
    return await masking_rule_service.list_rules(
        db=db,
        tenant_id=tenant.id,
        pii_type_name=pii_type_name,
    )


# ---------------------------------------------------------------------------
# GET /strategies -- list available masking strategies
# ---------------------------------------------------------------------------
@router.get(
    "/strategies",
    status_code=status.HTTP_200_OK,

    summary="List available masking strategies",
)
async def list_strategies(
    db: AsyncSession = Depends(get_db),
):
    """Return the full catalogue of supported masking strategies with
    descriptions and examples."""
    return await masking_rule_service.list_strategies(db=db)


# ---------------------------------------------------------------------------
# PATCH /{pii_type_name} -- set tenant masking strategy
# ---------------------------------------------------------------------------
@router.patch(
    "/{pii_type_name}",
    status_code=status.HTTP_200_OK,

    responses={
        404: {"model": ErrorResponse, "description": "PII type not found"},
    },
    summary="Set masking strategy for a PII type",
)
async def update_masking_rule(
    pii_type_name: str,
    body: MaskingRuleUpdate,
    db: AsyncSession = Depends(get_db),
    tenant: Tenant = Depends(get_current_tenant),
    current_user: User = Depends(get_current_user),
):
    """Override the masking strategy for a specific PII type within the tenant."""
    return await masking_rule_service.update_rule(
        db=db,
        tenant_id=tenant.id,
        pii_type_name=pii_type_name,
        strategy=body.strategy,
        strategy_config=body.strategy_config,
        updated_by=current_user.id,
    )


# ---------------------------------------------------------------------------
# POST /reset-defaults -- reset to system defaults
# ---------------------------------------------------------------------------
@router.post(
    "/reset-defaults",
    status_code=status.HTTP_200_OK,

    summary="Reset masking rules to system defaults",
    dependencies=[Depends(require_role("admin"))],
)
async def reset_defaults(
    db: AsyncSession = Depends(get_db),
    tenant: Tenant = Depends(get_current_tenant),
    current_user: User = Depends(get_current_user),
):
    """Remove all tenant masking overrides and revert to system defaults."""
    await masking_rule_service.reset_defaults(
        db=db,
        tenant_id=tenant.id,
        reset_by=current_user.id,
    )
    return SuccessResponse(message="Masking rules reset to system defaults.")
