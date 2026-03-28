import {
  UserRole,
  JobStatus,
  JobType,
  APIRequestStatus,
  NotificationType,
  PIITier,
  DetectionSource,
  MaskingStrategyType,
} from "./enums";

// ---- Tenant & User ----

export interface Tenant {
  id: string;
  name: string;
  slug: string;
  is_active: boolean;
  settings: Record<string, unknown>;
  created_at: string;
  updated_at: string;
}

export interface User {
  id: string;
  tenant_id: string;
  email: string;
  full_name: string;
  role: UserRole;
  is_active: boolean;
  last_login: string | null;
  created_at: string;
  updated_at: string;
}

export interface UserSession {
  user: User;
  tenant: Tenant;
  access_token: string;
  refresh_token: string;
  expires_at: string;
}

// ---- PII Taxonomy ----

export interface PIICategory {
  id: number;
  name: string;
  display_name: string;
  description?: string;
  tier: number;
  sort_order: number;
  color?: string;
  pii_types?: PIITypeCore[];
}

/** Core PII type fields as returned by the backend */
export interface PIITypeCore {
  id: number;
  name: string;
  display_name: string;
  category_id: number;
  category_name?: string;
  description?: string;
  gliner_aliases: string[];
  default_threshold: number;
  is_system: boolean;
  is_sensitive: boolean;
  compliance_tags: string[];
  category?: {
    id: number;
    name: string;
    display_name: string;
    tier: number;
  };
}

/** Tenant-specific config overlay */
export interface TenantPIIConfig {
  is_enabled: boolean;
  custom_threshold?: number | null;
  custom_aliases?: string[] | null;
  notes?: string | null;
}

/** Wrapper item returned by GET /api/v1/pii-types/ */
export interface PIITypeItem {
  pii_type: PIITypeCore;
  tenant_config: TenantPIIConfig | null;
}

/**
 * Flat view of a PII type for UI consumption.
 * Created by unwrapping PIITypeItem.
 */
export interface PIIType {
  id: number;
  name: string;
  display_name: string;
  category_id: number;
  category_name: string;
  tier: number;
  description: string;
  gliner_aliases: string[];
  default_threshold: number;
  compliance_tags: string[];
  is_system: boolean;
  is_sensitive: boolean;
  is_enabled: boolean;
  custom_threshold?: number | null;
}

export interface CustomPIIType {
  id: number;
  tenant_id: string;
  name: string;
  display_name: string;
  category_id: number;
  description?: string;
  gliner_aliases: string[];
  default_threshold: number;
  is_enabled: boolean;
  compliance_tags: string[];
  created_at: string;
}

// ---- Rules ----

export interface RegexRule {
  id: string;
  tenant_id: string | null;
  pii_type_name: string;
  pattern: string;
  description: string;
  priority: number;
  is_active: boolean;
  is_system: boolean;
  created_at: string;
  updated_at: string;
}

export interface FieldPattern {
  id: string;
  tenant_id: string | null;
  pii_type_name: string;
  pattern: string;
  description: string;
  is_active: boolean;
  is_system: boolean;
  created_at: string;
  updated_at: string;
}

export interface ContextRule {
  id: string;
  tenant_id: string | null;
  pii_type_name: string;
  positive_contexts: string[];
  negative_contexts: string[];
  boost_score: number;
  penalty_score: number;
  is_active: boolean;
  is_system: boolean;
  created_at: string;
  updated_at: string;
}

export interface MaskingStrategy {
  id: string;
  name: string;
  strategy_type: MaskingStrategyType;
  description: string;
  config: Record<string, unknown>;
}

export interface MaskingRule {
  id: string;
  tenant_id: string;
  pii_type_name: string;
  strategy_id: string;
  strategy_name: string;
  strategy_type: MaskingStrategyType;
  custom_config: Record<string, unknown>;
  created_at: string;
  updated_at: string;
}

// ---- Detection ----

export interface Detection {
  label: string;
  text: string;
  start: number;
  end: number;
  score: number;
  source: DetectionSource | string;
  instance_id?: string | null;
  replacement_tag?: string | null;
  meta?: Record<string, unknown>;
}

export interface DetectionStats {
  total: number;
  by_type: Record<string, number>;
  by_source: Record<string, number>;
}

export interface DetectionJob {
  id: string;
  tenant_id: string;
  user_id: string;
  job_type: JobType;
  status: JobStatus;
  input_text_preview: string;
  result_count: number;
  error_message: string | null;
  priority: number;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
}

export interface DetectResponse {
  detections: Detection[];
  stats?: DetectionStats;
  detection_count?: number;
  pii_types_found?: string[];
  processing_time_ms?: number;
  text_length?: number;
  job_id?: string;
}

export interface RedactResponse {
  redacted_text: string;
  detections: Detection[];
  stats?: DetectionStats;
  detection_count?: number;
  processing_time_ms?: number;
  job_id?: string;
}

// ---- Learning ----

export interface PendingRule {
  id: string;
  tenant_id: string;
  group_key: string;
  suggested_label: string;
  seen_count: number;
  example_contexts: string[];
  keywords: string[];
  avg_score: number;
  status: string;
  created_at: string;
  updated_at: string;
}

export interface TrainingExample {
  _id: string;
  id?: string;
  tenant_id: string;
  structure: string;
  label?: string;
  entity_type?: string;
  source: DetectionSource | string;
  score: number;
  keywords?: string[];
  co_labels?: string[];
  length?: number;
  metadata?: Record<string, unknown>;
  created_at: string;
}

// ---- Models ----

export interface ModelVersion {
  id: string | number;
  tenant_id: string;
  version: string | number;
  is_active: boolean;
  num_labels: number;
  num_examples: number;
  train_size?: number | null;
  val_size?: number | null;
  val_accuracy?: number | null;
  val_weighted_f1?: number | null;
  metrics_detail?: Record<string, unknown> | null;
  model_bucket_path?: string;
  vocab_bucket_path?: string;
  training_trigger?: string;
  trained_at: string;
  created_at?: string;
}

export interface ModelMetrics {
  overall?: {
    accuracy: number;
    f1_score: number;
    precision: number;
    recall: number;
  };
  per_label?: Record<
    string,
    {
      precision: number;
      recall: number;
      f1: number;
      f1_score?: number;
      support: number;
    }
  >;
  loss_history?: number[];
  metrics_detail?: Record<string, unknown>;
  // Flat fields returned by the metrics endpoint
  id?: number;
  version?: number;
  is_active?: boolean;
  num_labels?: number;
  num_examples?: number;
  train_size?: number | null;
  val_size?: number | null;
  val_accuracy?: number | null;
  val_weighted_f1?: number | null;
  training_trigger?: string;
  trained_at?: string;
}

// ---- API Management ----

export interface APIKey {
  id: string;
  tenant_id: string;
  name: string;
  key_prefix: string;
  scopes: string[];
  rate_limit: number;
  is_active: boolean;
  last_used_at: string | null;
  expires_at: string | null;
  created_at: string;
  created_by: string;
}

export interface APIAccessRequest {
  id: string;
  tenant_id: string;
  user_id: string;
  user_email: string;
  user_name: string;
  use_case: string;
  requested_scopes: string[];
  status: APIRequestStatus;
  reviewed_by: string | null;
  review_note: string | null;
  created_at: string;
  reviewed_at: string | null;
}

export interface APIUsageLog {
  id: string;
  api_key_id: string;
  endpoint: string;
  method: string;
  status_code: number;
  response_time_ms: number;
  request_size: number;
  response_size: number;
  created_at: string;
}

// ---- Audit & Notifications ----

export interface AuditLog {
  id: string;
  tenant_id: string;
  user_id: string;
  user_email: string;
  action: string;
  resource_type: string;
  resource_id: string | null;
  details: Record<string, unknown>;
  ip_address: string;
  created_at: string;
}

export interface Notification {
  id: string;
  tenant_id: string;
  user_id: string | null;
  type: NotificationType;
  title: string;
  message: string;
  is_read: boolean;
  action_url: string | null;
  created_at: string;
}
