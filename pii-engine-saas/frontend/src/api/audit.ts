import { get } from "./client";
import { AuditLog } from "@/types/models";
import { PaginatedResponse } from "@/types/api";
import apiClient from "./client";

export interface AuditLogFilters {
  action?: string;
  user_id?: string;
  resource_type?: string;
  date_from?: string;
  date_to?: string;
}

export const auditApi = {
  listLogs(
    filters: AuditLogFilters = {},
    page: number = 1,
    pageSize: number = 20
  ): Promise<PaginatedResponse<AuditLog>> {
    return get("/audit/logs", {
      params: {
        ...filters,
        page,
        page_size: pageSize,
      },
    });
  },

  async exportLogs(filters: AuditLogFilters = {}): Promise<Blob> {
    const response = await apiClient.get("/audit/logs/export", {
      params: filters,
      responseType: "blob",
    });
    return response.data;
  },
};
