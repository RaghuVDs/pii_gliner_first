import { get, post, patch } from "./client";
import { APIKey, APIAccessRequest, APIUsageLog } from "@/types/models";
import { PaginatedResponse, SuccessResponse } from "@/types/api";
import { APIRequestStatus } from "@/types/enums";

export interface CreateKeyRequest {
  name: string;
  scopes: string[];
  rate_limit?: number;
  expires_in_days?: number;
}

export interface ReviewRequestData {
  status: APIRequestStatus.APPROVED | APIRequestStatus.DENIED;
  review_note?: string;
}

export interface UsageAnalytics {
  total_requests: number;
  total_errors: number;
  avg_response_time_ms: number;
  requests_by_endpoint: Record<string, number>;
  requests_by_day: { date: string; count: number }[];
}

export const apiKeysApi = {
  listKeys(): Promise<APIKey[]> {
    return get("/api/keys");
  },

  createKey(
    data: CreateKeyRequest
  ): Promise<APIKey & { secret_key: string }> {
    return post("/api/keys", data);
  },

  revokeKey(keyId: string): Promise<SuccessResponse> {
    return post(`/api/keys/${keyId}/revoke`);
  },

  rotateKey(keyId: string): Promise<APIKey & { secret_key: string }> {
    return post(`/api/keys/${keyId}/rotate`);
  },

  getKeyUsage(keyId: string): Promise<PaginatedResponse<APIUsageLog>> {
    return get(`/api/keys/${keyId}/usage`);
  },

  listAccessRequests(
    status?: APIRequestStatus
  ): Promise<APIAccessRequest[]> {
    return get("/api/requests", { params: { status } });
  },

  createAccessRequest(data: {
    use_case: string;
    requested_scopes: string[];
  }): Promise<APIAccessRequest> {
    return post("/api/requests", data);
  },

  reviewAccessRequest(
    requestId: string,
    data: ReviewRequestData
  ): Promise<APIAccessRequest> {
    return patch(`/api/requests/${requestId}`, data);
  },

  getUsageAnalytics(
    dateFrom?: string,
    dateTo?: string
  ): Promise<UsageAnalytics> {
    return get("/api/usage", {
      params: { date_from: dateFrom, date_to: dateTo },
    });
  },
};
