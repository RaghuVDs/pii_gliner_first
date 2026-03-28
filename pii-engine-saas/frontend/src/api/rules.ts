import { get, post, patch, del } from "./client";
import {
  RegexRule,
  FieldPattern,
  ContextRule,
  MaskingRule,
  MaskingStrategy,
} from "@/types/models";
import { SuccessResponse } from "@/types/api";

export const rulesApi = {
  // ---- Regex Rules ----
  listRegexRules(piiTypeName?: string): Promise<RegexRule[]> {
    return get("/regex-rules/", { params: { pii_type_name: piiTypeName } });
  },

  createRegexRule(data: Partial<RegexRule>): Promise<RegexRule> {
    return post("/regex-rules/", data);
  },

  updateRegexRule(id: string, data: Partial<RegexRule>): Promise<RegexRule> {
    return patch(`/regex-rules/${id}`, data);
  },

  deleteRegexRule(id: string): Promise<SuccessResponse> {
    return del(`/regex-rules/${id}`);
  },

  testRegexPattern(
    pattern: string,
    testText: string
  ): Promise<{ matches: { text: string; start: number; end: number }[] }> {
    return post("/regex-rules/test", { pattern, test_text: testText });
  },

  // ---- Field Patterns ----
  listFieldPatterns(piiTypeName?: string): Promise<FieldPattern[]> {
    return get("/field-patterns/", {
      params: { pii_type_name: piiTypeName },
    });
  },

  createFieldPattern(data: Partial<FieldPattern>): Promise<FieldPattern> {
    return post("/field-patterns/", data);
  },

  updateFieldPattern(
    id: string,
    data: Partial<FieldPattern>
  ): Promise<FieldPattern> {
    return patch(`/field-patterns/${id}`, data);
  },

  deleteFieldPattern(id: string): Promise<SuccessResponse> {
    return del(`/field-patterns/${id}`);
  },

  // ---- Context Rules ----
  listContextRules(piiTypeName?: string): Promise<ContextRule[]> {
    return get("/context-rules/", { params: { pii_type_name: piiTypeName } });
  },

  createContextRule(data: Partial<ContextRule>): Promise<ContextRule> {
    return post("/context-rules/", data);
  },

  updateContextRule(
    id: string,
    data: Partial<ContextRule>
  ): Promise<ContextRule> {
    return patch(`/context-rules/${id}`, data);
  },

  deleteContextRule(id: string): Promise<SuccessResponse> {
    return del(`/context-rules/${id}`);
  },

  // ---- Masking ----
  listMaskingRules(): Promise<MaskingRule[]> {
    return get("/masking-rules/");
  },

  listMaskingStrategies(): Promise<MaskingStrategy[]> {
    return get("/masking-rules/strategies");
  },

  updateMaskingRule(
    piiTypeName: string,
    data: Partial<MaskingRule>
  ): Promise<MaskingRule> {
    return patch(`/masking-rules/${piiTypeName}`, data);
  },
};
