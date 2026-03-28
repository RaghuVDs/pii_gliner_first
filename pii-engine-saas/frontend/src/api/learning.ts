import { get, post, del } from "./client";
import { PendingRule, TrainingExample } from "@/types/models";
import { PaginatedResponse, SuccessResponse } from "@/types/api";

export interface PendingRuleStats {
  total: number;
  pending: number;
  promoted: number;
  rejected: number;
  avg_seen_count: number;
}

export interface TrainingStats {
  total_examples: number;
  total?: number;
  by_label: Record<string, number>;
  by_source: Record<string, number>;
}

export const learningApi = {
  listPendingRules(
    page: number = 1,
    pageSize: number = 20
  ): Promise<PaginatedResponse<PendingRule>> {
    return get("/learning/pending-rules", {
      params: { page, page_size: pageSize },
    });
  },

  getPendingStats(): Promise<PendingRuleStats> {
    return get("/learning/pending-rules/stats");
  },

  promoteRule(
    ruleId: string,
    label: string,
    keywords?: string[]
  ): Promise<SuccessResponse> {
    return post(`/learning/pending-rules/${ruleId}/promote`, {
      label,
      keywords,
    });
  },

  rejectRule(ruleId: string): Promise<SuccessResponse> {
    return post(`/learning/pending-rules/${ruleId}/reject`);
  },

  autoPromote(threshold: number): Promise<{ promoted_count: number }> {
    return post("/learning/pending-rules/auto-promote", { threshold });
  },

  listTrainingData(
    page: number = 1,
    pageSize: number = 20,
    label?: string
  ): Promise<PaginatedResponse<TrainingExample>> {
    return get("/learning/training-data", {
      params: { page, page_size: pageSize, entity_type: label },
    });
  },

  getTrainingStats(): Promise<TrainingStats> {
    return get("/learning/training-data/stats");
  },

  deleteTrainingExample(id: string): Promise<SuccessResponse> {
    return del(`/learning/training-data/${id}`);
  },
};
