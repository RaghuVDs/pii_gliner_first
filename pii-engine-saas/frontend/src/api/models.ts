import { get, post } from "./client";
import { ModelVersion, ModelMetrics } from "@/types/models";
import { SuccessResponse } from "@/types/api";

export const modelsApi = {
  listVersions(): Promise<ModelVersion[]> {
    return get("/models/");
  },

  getActiveModel(): Promise<ModelVersion> {
    return get("/models/active");
  },

  triggerRetrain(epochs?: number): Promise<{ job_id: string }> {
    return post("/models/retrain", { epochs });
  },

  getRetrainStatus(jobId: string): Promise<{
    job_id: string;
    status: string;
    progress: number;
    message: string;
  }> {
    return get(`/models/retrain/${jobId}`);
  },

  activateVersion(versionId: string): Promise<SuccessResponse> {
    return post(`/models/${versionId}/activate`);
  },

  getMetrics(): Promise<ModelMetrics> {
    return get("/models/metrics");
  },

  getVersionMetrics(versionId: string): Promise<ModelMetrics> {
    return get(`/models/metrics/${versionId}`);
  },
};
