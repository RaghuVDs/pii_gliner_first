"""
Analytics and reporting endpoints.

Provides dashboard overview stats, time-series data, cross-tabulations,
and heatmap data for PII detections.  All queries are scoped to the
current tenant and support configurable date ranges.
"""

from __future__ import annotations
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db

from datetime import date, datetime
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, Query, status
from pydantic import BaseModel, ConfigDict, Field

from app.api.v1.dependencies import (
    get_current_tenant,
    get_current_user,
)
from app.models.tenant import Tenant
from app.models.user import User
from app.services import analytics_service

router = APIRouter(tags=["Analytics"])


# ---------------------------------------------------------------------------
# Inline response schemas
# ---------------------------------------------------------------------------

class OverviewStats(BaseModel):
    total_scans: int = 0
    total_entities_detected: int = 0
    unique_pii_types_detected: int = 0
    avg_entities_per_scan: float = 0.0
    scans_today: int = 0
    scans_this_week: int = 0
    scans_this_month: int = 0
    active_users: int = 0
    active_api_keys: int = 0


class DetectionsByType(BaseModel):
    pii_type: str
    count: int
    percentage: float = 0.0


class DetectionsBySource(BaseModel):
    source: str
    count: int
    percentage: float = 0.0


class DetectionCrossTab(BaseModel):
    pii_type: str
    source: str
    count: int


class TimelinePoint(BaseModel):
    timestamp: str
    count: int
    pii_type: str | None = None


class HeatmapCell(BaseModel):
    pii_type: str
    period: str
    count: int


class TopPIIType(BaseModel):
    pii_type: str
    count: int
    trend: float = Field(0.0, description="Percentage change vs previous period")


class SourceEffectiveness(BaseModel):
    source: str
    total_detections: int
    avg_confidence: float
    unique_types_detected: int
    false_positive_rate: float | None = None


# ---------------------------------------------------------------------------
# Helper: common date range parameters
# ---------------------------------------------------------------------------
def _date_range_params(
    start_date: date | None = Query(None, description="Start date (inclusive)"),
    end_date: date | None = Query(None, description="End date (inclusive)"),
):
    return {"start_date": start_date, "end_date": end_date}


# ---------------------------------------------------------------------------
# GET /overview -- dashboard overview stats
# ---------------------------------------------------------------------------
@router.get(
    "/overview",
    status_code=status.HTTP_200_OK,

    summary="Dashboard overview stats",
)
async def overview(
    db: AsyncSession = Depends(get_db),
    tenant: Tenant = Depends(get_current_tenant),
):
    """Return high-level dashboard statistics for the current tenant."""
    return await analytics_service.get_overview(db=db, tenant_id=tenant.id)


# ---------------------------------------------------------------------------
# GET /detections/by-type -- detection counts by PII type
# ---------------------------------------------------------------------------
@router.get(
    "/detections/by-type",
    status_code=status.HTTP_200_OK,

    summary="Detections by PII type",
)
async def detections_by_type(
    db: AsyncSession = Depends(get_db),
    tenant: Tenant = Depends(get_current_tenant),
    start_date: date | None = Query(None, description="Start date (inclusive)"),
    end_date: date | None = Query(None, description="End date (inclusive)"),
    limit: int = Query(20, ge=1, le=100),
):
    """Return detection counts grouped by PII type within an optional
    date range."""
    return await analytics_service.detections_by_type(db=db, 
        tenant_id=tenant.id,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
    )


# ---------------------------------------------------------------------------
# GET /detections/by-source -- detection counts by source
# ---------------------------------------------------------------------------
@router.get(
    "/detections/by-source",
    status_code=status.HTTP_200_OK,

    summary="Detections by source",
)
async def detections_by_source(
    db: AsyncSession = Depends(get_db),
    tenant: Tenant = Depends(get_current_tenant),
    start_date: date | None = Query(None),
    end_date: date | None = Query(None),
):
    """Return detection counts grouped by detection source (gliner, regex,
    field_pattern, context)."""
    return await analytics_service.detections_by_source(db=db, 
        tenant_id=tenant.id,
        start_date=start_date,
        end_date=end_date,
    )


# ---------------------------------------------------------------------------
# GET /detections/by-type-source -- cross-tabulation
# ---------------------------------------------------------------------------
@router.get(
    "/detections/by-type-source",
    status_code=status.HTTP_200_OK,

    summary="Detections cross-tab (type x source)",
)
async def detections_by_type_source(
    db: AsyncSession = Depends(get_db),
    tenant: Tenant = Depends(get_current_tenant),
    start_date: date | None = Query(None),
    end_date: date | None = Query(None),
):
    """Return a cross-tabulation of detections by PII type and source."""
    return await analytics_service.detections_by_type_source(db=db, 
        tenant_id=tenant.id,
        start_date=start_date,
        end_date=end_date,
    )


# ---------------------------------------------------------------------------
# GET /detections/timeline -- time series of detections
# ---------------------------------------------------------------------------
@router.get(
    "/detections/timeline",
    status_code=status.HTTP_200_OK,

    summary="Detection timeline",
)
async def detections_timeline(
    db: AsyncSession = Depends(get_db),
    tenant: Tenant = Depends(get_current_tenant),
    start_date: date | None = Query(None),
    end_date: date | None = Query(None),
    granularity: str = Query(
        "day", description="Time bucket granularity: hour, day, week, month",
    ),
    pii_type: str | None = Query(None, description="Filter by specific PII type"),
):
    """Return a time-series of detection counts, optionally filtered by
    PII type and grouped by the requested granularity."""
    return await analytics_service.detections_timeline(db=db, 
        tenant_id=tenant.id,
        start_date=start_date,
        end_date=end_date,
        granularity=granularity,
        pii_type=pii_type,
    )


# ---------------------------------------------------------------------------
# GET /detections/heatmap -- PII type frequency heatmap data
# ---------------------------------------------------------------------------
@router.get(
    "/detections/heatmap",
    status_code=status.HTTP_200_OK,

    summary="Detection heatmap data",
)
async def detections_heatmap(
    db: AsyncSession = Depends(get_db),
    tenant: Tenant = Depends(get_current_tenant),
    start_date: date | None = Query(None),
    end_date: date | None = Query(None),
    granularity: str = Query("day", description="Period granularity: hour, day, week"),
):
    """Return PII-type-by-period frequency data suitable for rendering a
    heatmap visualisation."""
    return await analytics_service.detections_heatmap(db=db, 
        tenant_id=tenant.id,
        start_date=start_date,
        end_date=end_date,
        granularity=granularity,
    )


# ---------------------------------------------------------------------------
# GET /top-pii-types -- top N most detected PII types
# ---------------------------------------------------------------------------
@router.get(
    "/top-pii-types",
    status_code=status.HTTP_200_OK,

    summary="Top PII types",
)
async def top_pii_types(
    db: AsyncSession = Depends(get_db),
    tenant: Tenant = Depends(get_current_tenant),
    start_date: date | None = Query(None),
    end_date: date | None = Query(None),
    limit: int = Query(10, ge=1, le=50),
):
    """Return the top N most-detected PII types with trend data comparing
    to the previous equivalent period."""
    return await analytics_service.top_pii_types(db=db, 
        tenant_id=tenant.id,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
    )


# ---------------------------------------------------------------------------
# GET /source-effectiveness -- comparison of detection sources
# ---------------------------------------------------------------------------
@router.get(
    "/source-effectiveness",
    status_code=status.HTTP_200_OK,

    summary="Source effectiveness comparison",
)
async def source_effectiveness(
    db: AsyncSession = Depends(get_db),
    tenant: Tenant = Depends(get_current_tenant),
    start_date: date | None = Query(None),
    end_date: date | None = Query(None),
):
    """Compare detection sources (gliner, regex, etc.) by volume,
    confidence, type coverage, and estimated false-positive rate."""
    return await analytics_service.source_effectiveness(db=db, 
        tenant_id=tenant.id,
        start_date=start_date,
        end_date=end_date,
    )
