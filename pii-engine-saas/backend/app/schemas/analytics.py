"""Analytics and dashboard schemas."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Overview / KPIs
# ---------------------------------------------------------------------------
class OverviewResponse(BaseModel):
    """High-level KPI snapshot for the analytics dashboard."""

    total_scans: int = Field(..., ge=0)
    total_pii_found: int = Field(..., ge=0)
    unique_pii_types: int = Field(..., ge=0)
    active_rules: int = Field(..., ge=0)
    accuracy_estimate: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Estimated detection accuracy (0-1), None if insufficient data",
    )
    scans_today: int = Field(..., ge=0)
    scans_this_week: int = Field(..., ge=0)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "total_scans": 12450,
                "total_pii_found": 34200,
                "unique_pii_types": 18,
                "active_rules": 42,
                "accuracy_estimate": 0.94,
                "scans_today": 85,
                "scans_this_week": 620,
            }
        },
    )


# ---------------------------------------------------------------------------
# Breakdown by PII type
# ---------------------------------------------------------------------------
class ByTypeItem(BaseModel):
    """A single row in the by-type breakdown."""

    label: str
    count: int = Field(..., ge=0)
    percentage: float = Field(..., ge=0.0, le=100.0)


class ByTypeResponse(BaseModel):
    """PII detection counts grouped by type/label."""

    items: list[ByTypeItem] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Breakdown by detection source
# ---------------------------------------------------------------------------
class BySourceItem(BaseModel):
    """A single row in the by-source breakdown."""

    source: str
    count: int = Field(..., ge=0)
    percentage: float = Field(..., ge=0.0, le=100.0)


class BySourceResponse(BaseModel):
    """PII detection counts grouped by source engine."""

    items: list[BySourceItem] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Timeline
# ---------------------------------------------------------------------------
class TimelineDataPoint(BaseModel):
    """A single data point on the timeline chart."""

    date: str = Field(..., description="ISO date string (YYYY-MM-DD)")
    count: int = Field(..., ge=0)
    by_label: dict[str, int] | None = Field(
        None,
        description="Optional per-label breakdown for this date",
    )


class TimelineResponse(BaseModel):
    """Time-series detection data for charting."""

    data: list[TimelineDataPoint] = Field(default_factory=list)
    period: str = Field(
        ...,
        description="Aggregation period: day, week, month",
    )


# ---------------------------------------------------------------------------
# Heatmap
# ---------------------------------------------------------------------------
class HeatmapCell(BaseModel):
    """A single cell in the label-by-hour heatmap."""

    label: str
    hour: int = Field(..., ge=0, le=23)
    count: int = Field(..., ge=0)


class HeatmapResponse(BaseModel):
    """Heatmap of detections by PII label and hour-of-day."""

    cells: list[HeatmapCell] = Field(default_factory=list)
