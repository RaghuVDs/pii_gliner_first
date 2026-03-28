import { get, post } from "./client";
import { DetectResponse, RedactResponse, DetectionJob } from "@/types/models";
import { PaginatedResponse, SuccessResponse } from "@/types/api";
import { JobStatus } from "@/types/enums";

export const detectionApi = {
  detect(
    text: string,
    piiTypes?: string[]
  ): Promise<DetectResponse> {
    return post("/detection/detect", { text, pii_types: piiTypes });
  },

  redact(
    text: string,
    piiTypes?: string[]
  ): Promise<RedactResponse> {
    return post("/detection/redact", { text, pii_types: piiTypes });
  },

  detectAsync(
    text: string,
    piiTypes?: string[],
    priority?: number
  ): Promise<{ job_id: string }> {
    return post("/detection/detect-async", {
      text,
      pii_types: piiTypes,
      priority,
    });
  },

  listJobs(
    page: number = 1,
    pageSize: number = 20,
    status?: JobStatus
  ): Promise<PaginatedResponse<DetectionJob>> {
    return get("/detection/jobs", {
      params: { page, page_size: pageSize, status },
    });
  },

  getJob(jobId: string): Promise<DetectionJob> {
    return get(`/detection/jobs/${jobId}`);
  },

  cancelJob(jobId: string): Promise<SuccessResponse> {
    return post(`/detection/jobs/${jobId}/cancel`);
  },
};
