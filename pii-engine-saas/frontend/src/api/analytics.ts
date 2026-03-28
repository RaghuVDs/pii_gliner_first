import { get } from "./client";

export interface OverviewData {
  total_scans: number;
  total_pii_found: number;
  active_rules: number;
  model_accuracy: number;
  scans_today: number;
  pii_found_today: number;
}

export interface TypeBreakdown {
  pii_type: string;
  count: number;
  percentage: number;
}

export interface SourceBreakdown {
  source: string;
  count: number;
  percentage: number;
}

export interface TimelinePoint {
  timestamp: string;
  scans: number;
  detections: number;
}

export interface HeatmapCell {
  hour: number;
  day_of_week: number;
  count: number;
}

export interface SourceEffectiveness {
  source: string;
  avg_confidence: number;
  total_detections: number;
  unique_types: number;
}

export const analyticsApi = {
  getOverview(): Promise<OverviewData> {
    return get("/analytics/overview");
  },

  getByType(dateFrom?: string, dateTo?: string): Promise<TypeBreakdown[]> {
    return get("/analytics/detections/by-type", {
      params: { date_from: dateFrom, date_to: dateTo },
    });
  },

  getBySource(
    dateFrom?: string,
    dateTo?: string
  ): Promise<SourceBreakdown[]> {
    return get("/analytics/detections/by-source", {
      params: { date_from: dateFrom, date_to: dateTo },
    });
  },

  getTimeline(
    dateFrom?: string,
    dateTo?: string,
    interval?: string
  ): Promise<TimelinePoint[]> {
    return get("/analytics/detections/timeline", {
      params: { date_from: dateFrom, date_to: dateTo, interval },
    });
  },

  getHeatmap(dateFrom?: string, dateTo?: string): Promise<HeatmapCell[]> {
    return get("/analytics/detections/heatmap", {
      params: { date_from: dateFrom, date_to: dateTo },
    });
  },

  getTopTypes(
    n?: number,
    dateFrom?: string,
    dateTo?: string
  ): Promise<TypeBreakdown[]> {
    return get("/analytics/top-pii-types", {
      params: { n, date_from: dateFrom, date_to: dateTo },
    });
  },

  getSourceEffectiveness(): Promise<SourceEffectiveness[]> {
    return get("/analytics/source-effectiveness");
  },
};
