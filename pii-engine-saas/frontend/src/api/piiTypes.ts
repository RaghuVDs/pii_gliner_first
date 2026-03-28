import { get, post, patch, del } from "./client";
import {
  PIICategory,
  PIIType,
  PIITypeItem,
  TenantPIIConfig,
  CustomPIIType,
} from "@/types/models";
import { PaginatedResponse, SuccessResponse } from "@/types/api";

export interface ListPIITypesParams {
  category?: string;
  tier?: number;
  enabled?: boolean;
  search?: string;
  page?: number;
  pageSize?: number;
}

/**
 * Unwrap a PIITypeItem (backend wrapper) into a flat PIIType for UI use.
 */
function unwrapPIITypeItem(item: PIITypeItem): PIIType {
  const pt = item.pii_type;
  const tc = item.tenant_config;
  return {
    id: pt.id,
    name: pt.name,
    display_name: pt.display_name,
    category_id: pt.category_id,
    category_name: pt.category_name || pt.category?.name || "",
    tier: pt.category?.tier ?? 0,
    description: pt.description || "",
    gliner_aliases: pt.gliner_aliases || [],
    default_threshold: tc?.custom_threshold ?? pt.default_threshold,
    compliance_tags: pt.compliance_tags || [],
    is_system: pt.is_system,
    is_sensitive: pt.is_sensitive,
    is_enabled: tc?.is_enabled ?? true,
    custom_threshold: tc?.custom_threshold,
  };
}

export const piiTypesApi = {
  listCategories(): Promise<PIICategory[]> {
    return get("/pii-types/categories");
  },

  async listPIITypes(
    params: ListPIITypesParams = {}
  ): Promise<PaginatedResponse<PIIType>> {
    // The backend returns items wrapped as { pii_type, tenant_config }.
    // We unwrap them into flat PIIType objects for the UI.
    const raw = await get<PaginatedResponse<PIITypeItem | PIIType>>("/pii-types", {
      params: {
        category: params.category,
        tier: params.tier,
        enabled: params.enabled,
        search: params.search,
        page: params.page || 1,
        page_size: params.pageSize || 50,
      },
    });

    const items: PIIType[] = (raw.items || []).map((item: PIITypeItem | PIIType) => {
      // Handle wrapped format: { pii_type: {...}, tenant_config: {...} }
      if ("pii_type" in item) {
        return unwrapPIITypeItem(item as PIITypeItem);
      }
      // Already flat (fallback)
      return item as PIIType;
    });

    return { ...raw, items };
  },

  getPIIType(id: number | string): Promise<PIIType> {
    return get(`/pii-types/${id}`);
  },

  updateConfig(
    id: number | string,
    config: Partial<TenantPIIConfig>
  ): Promise<TenantPIIConfig> {
    return patch(`/pii-types/${id}/config`, config);
  },

  batchConfig(
    piiTypeIds: (number | string)[],
    isEnabled: boolean
  ): Promise<SuccessResponse> {
    return post("/pii-types/batch-config", {
      pii_type_ids: piiTypeIds,
      is_enabled: isEnabled,
    });
  },

  createCustomType(data: Partial<CustomPIIType>): Promise<CustomPIIType> {
    return post("/pii-types/custom", data);
  },

  updateCustomType(
    id: number | string,
    data: Partial<CustomPIIType>
  ): Promise<CustomPIIType> {
    return patch(`/pii-types/custom/${id}`, data);
  },

  deleteCustomType(id: number | string): Promise<SuccessResponse> {
    return del(`/pii-types/custom/${id}`);
  },

  resetDefaults(): Promise<SuccessResponse> {
    return post("/pii-types/reset-defaults");
  },
};
