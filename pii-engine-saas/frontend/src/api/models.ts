import { get, post } from "./client";
import { ModelVersion, ModelMetrics } from "@/types/models";
import { SuccessResponse } from "@/types/api";

export interface InlineRetrainResult {
  status: string;
  reason?: string;
  num_examples?: number;
  train_size?: number;
  val_size?: number;
  num_labels?: number;
  val_accuracy?: number;
  val_weighted_f1?: number;
  stopped_epoch?: number;
  best_epoch?: number;
  early_stopped?: boolean;
  best_val_loss?: number;
  val_per_label?: Record<string, Record<string, number>>;
  version_id?: number;
  version_number?: number;
  timestamp?: string;
}

export const modelsApi = {
  listVersions(): Promise<ModelVersion[]> {
    return get("/models/");
  },

  getActiveModel(): Promise<ModelVersion> {
    return get("/models/active");
  },

  triggerRetrain(epochs?: number): Promise<InlineRetrainResult> {
    // Training runs inline and may take several minutes - use extended timeout
    return post("/models/retrain", { epochs }, { timeout: 600000 });
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

  getMetrics(): Promise<ModelMetrics[]> {
    return get("/models/metrics");
  },

  getVersionMetrics(versionId: string): Promise<ModelMetrics> {
    return get(`/models/metrics/${versionId}`);
  },
};
