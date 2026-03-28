export enum UserRole {
  SUPER_ADMIN = "super_admin",
  TENANT_ADMIN = "tenant_admin",
  ANALYST = "analyst",
  VIEWER = "viewer",
  API_USER = "api_user",
}

export enum JobStatus {
  PENDING = "pending",
  PROCESSING = "processing",
  COMPLETED = "completed",
  FAILED = "failed",
  CANCELLED = "cancelled",
}

export enum JobType {
  DETECTION = "detection",
  REDACTION = "redaction",
  BATCH_DETECTION = "batch_detection",
  MODEL_RETRAIN = "model_retrain",
}

export enum APIRequestStatus {
  PENDING = "pending",
  APPROVED = "approved",
  DENIED = "denied",
  REVOKED = "revoked",
}

export enum NotificationType {
  INFO = "info",
  WARNING = "warning",
  ERROR = "error",
  SUCCESS = "success",
}

export enum PIITier {
  TIER_1 = "tier_1",
  TIER_2 = "tier_2",
  TIER_3 = "tier_3",
}

export enum DetectionSource {
  GLINER = "gliner",
  REGEX = "regex",
  CONTEXT = "context",
  LSTM = "lstm",
  FIELD_PATTERN = "field_pattern",
}

export enum MaskingStrategyType {
  REDACT = "redact",
  MASK = "mask",
  HASH = "hash",
  ENCRYPT = "encrypt",
  TOKENIZE = "tokenize",
  GENERALIZE = "generalize",
  PLACEHOLDER = "placeholder",
}
