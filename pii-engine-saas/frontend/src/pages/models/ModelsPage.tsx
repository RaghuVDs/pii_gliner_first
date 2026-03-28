import React, { useEffect, useState, useCallback } from "react";
import {
  Card,
  Table,
  Button,
  Space,
  Typography,
  Tag,
  InputNumber,
  Row,
  Col,
  Statistic,
  message,
  Spin,
  Modal,
  Progress,
  Divider,
  Alert,
  Descriptions,
} from "antd";
import {
  RobotOutlined,
  ThunderboltOutlined,
  CheckCircleOutlined,
  SyncOutlined,
  SettingOutlined,
} from "@ant-design/icons";
import { Line } from "@ant-design/charts";
import { modelsApi, InlineRetrainResult } from "@/api/models";
import { ModelVersion, ModelMetrics } from "@/types/models";
import dayjs from "dayjs";

const { Title, Text } = Typography;

const AUTO_RETRAIN_KEY = "pii_auto_retrain_threshold";

const ModelsPage: React.FC = () => {
  const [activeModel, setActiveModel] = useState<ModelVersion | null>(null);
  const [versions, setVersions] = useState<ModelVersion[]>([]);
  const [metricsHistory, setMetricsHistory] = useState<ModelMetrics[]>([]);
  const [loading, setLoading] = useState(true);
  const [retrainEpochs, setRetrainEpochs] = useState<number>(50);
  const [retraining, setRetraining] = useState(false);
  const [retrainResult, setRetrainResult] = useState<InlineRetrainResult | null>(null);

  // Training Config
  const [autoRetrainThreshold, setAutoRetrainThreshold] = useState<number>(() => {
    const stored = localStorage.getItem(AUTO_RETRAIN_KEY);
    return stored ? parseInt(stored, 10) : 100;
  });

  const saveAutoRetrain = (val: number) => {
    setAutoRetrainThreshold(val);
    localStorage.setItem(AUTO_RETRAIN_KEY, String(val));
    message.success(`Auto-retrain threshold set to ${val} detections`);
  };

  const fetchData = useCallback(async () => {
    setLoading(true);
    try {
      const results = await Promise.allSettled([
        modelsApi.getActiveModel(),
        modelsApi.listVersions(),
        modelsApi.getMetrics(),
      ]);

      if (results[0].status === "fulfilled") {
        setActiveModel(results[0].value);
      }
      if (results[1].status === "fulfilled") {
        setVersions(results[1].value);
      }
      if (results[2].status === "fulfilled") {
        const metricsData = results[2].value;
        // The metrics endpoint returns an array of metric entries
        setMetricsHistory(Array.isArray(metricsData) ? metricsData : [metricsData]);
      }
    } catch {
      message.error("Failed to load model data");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  const handleRetrain = async () => {
    setRetraining(true);
    setRetrainResult(null);
    try {
      const result = await modelsApi.triggerRetrain(retrainEpochs);
      setRetrainResult(result);
      if (result.status === "trained") {
        message.success(
          `Model trained successfully! Val F1: ${((result.val_weighted_f1 || 0) * 100).toFixed(1)}%`
        );
        fetchData();
      } else if (result.status === "skipped") {
        message.warning(result.reason || "Training skipped - not enough data");
      } else {
        message.info(`Training result: ${result.status}`);
      }
    } catch {
      setRetrainResult(null);
      message.error("Failed to start retraining");
    } finally {
      setRetraining(false);
    }
  };

  const handleActivate = async (versionId: string | number) => {
    Modal.confirm({
      title: "Activate Model Version",
      content: "This will replace the currently active model. Continue?",
      onOk: async () => {
        try {
          await modelsApi.activateVersion(String(versionId));
          message.success("Model version activated");
          fetchData();
        } catch {
          message.error("Failed to activate model version");
        }
      },
    });
  };

  // Build loss chart data from the latest metrics entry that has history
  const latestWithMetrics = metricsHistory.find(
    (m) => m.metrics_detail && (m.metrics_detail as Record<string, unknown>).history_sample
  );
  const historySample = latestWithMetrics?.metrics_detail?.history_sample as
    | { val_loss?: number[]; train_loss?: number[] }
    | undefined;

  const lossChartData: { epoch: number; loss: number; type: string }[] = [];
  if (historySample?.train_loss) {
    historySample.train_loss.forEach((loss, i) => {
      lossChartData.push({ epoch: i + 1, loss, type: "Train" });
    });
  }
  if (historySample?.val_loss) {
    historySample.val_loss.forEach((loss, i) => {
      lossChartData.push({ epoch: i + 1, loss, type: "Validation" });
    });
  }

  const lossChartConfig = {
    data: lossChartData,
    xField: "epoch",
    yField: "loss",
    seriesField: "type",
    smooth: true,
    height: 250,
    color: ["#1677ff", "#ff4d4f"],
  };

  // Build per-label metrics from latest active version or latest metrics
  const latestPerLabel = latestWithMetrics?.metrics_detail?.val_per_label as
    | Record<string, { precision: number; recall: number; f1: number; support: number }>
    | undefined;

  const versionColumns = [
    {
      title: "Version",
      dataIndex: "version",
      key: "version",
      render: (version: string | number, record: ModelVersion) => (
        <Space>
          <Text strong>v{version}</Text>
          {record.is_active && <Tag color="green">ACTIVE</Tag>}
        </Space>
      ),
    },
    {
      title: "Accuracy",
      dataIndex: "val_accuracy",
      key: "accuracy",
      render: (v: number | null) =>
        v != null ? `${(Number(v) * 100).toFixed(1)}%` : "-",
    },
    {
      title: "F1 Score",
      dataIndex: "val_weighted_f1",
      key: "f1_score",
      render: (v: number | null) =>
        v != null ? `${(Number(v) * 100).toFixed(1)}%` : "-",
    },
    {
      title: "Labels",
      dataIndex: "num_labels",
      key: "labels_count",
    },
    {
      title: "Training Data",
      dataIndex: "num_examples",
      key: "training_examples_count",
    },
    {
      title: "Train/Val Size",
      key: "split",
      render: (_: unknown, record: ModelVersion) =>
        record.train_size && record.val_size
          ? `${record.train_size} / ${record.val_size}`
          : "-",
    },
    {
      title: "Trigger",
      dataIndex: "training_trigger",
      key: "trigger",
      render: (trigger: string) => (
        <Tag color={trigger === "manual" ? "blue" : "orange"}>
          {trigger || "auto"}
        </Tag>
      ),
    },
    {
      title: "Trained",
      dataIndex: "trained_at",
      key: "trained_at",
      render: (date: string) =>
        date ? dayjs(date).format("MMM DD, YYYY HH:mm") : "-",
    },
    {
      title: "Actions",
      key: "actions",
      render: (_: unknown, record: ModelVersion) =>
        record.is_active ? null : (
          <Button
            size="small"
            onClick={() => handleActivate(record.id)}
          >
            Activate
          </Button>
        ),
    },
  ];

  if (loading) {
    return (
      <div style={{ textAlign: "center", padding: 80 }}>
        <Spin size="large" />
      </div>
    );
  }

  return (
    <div>
      <Title level={4} style={{ marginBottom: 24 }}>
        ML Models
      </Title>

      {/* Active Model Card */}
      {activeModel && (
        <Card
          title={
            <Space>
              <RobotOutlined />
              Active Model
              <Tag color="green">v{activeModel.version}</Tag>
            </Space>
          }
          style={{ marginBottom: 24 }}
        >
          <Row gutter={24}>
            <Col span={6}>
              <Statistic
                title="Accuracy"
                value={
                  activeModel.val_accuracy != null
                    ? (Number(activeModel.val_accuracy) * 100).toFixed(1)
                    : "N/A"
                }
                suffix={activeModel.val_accuracy != null ? "%" : ""}
                prefix={<CheckCircleOutlined />}
                valueStyle={{ color: "#3f8600" }}
              />
            </Col>
            <Col span={6}>
              <Statistic
                title="F1 Score"
                value={
                  activeModel.val_weighted_f1 != null
                    ? (Number(activeModel.val_weighted_f1) * 100).toFixed(1)
                    : "N/A"
                }
                suffix={activeModel.val_weighted_f1 != null ? "%" : ""}
              />
            </Col>
            <Col span={6}>
              <Statistic
                title="Labels"
                value={activeModel.num_labels}
              />
            </Col>
            <Col span={6}>
              <Statistic
                title="Trained"
                value={
                  activeModel.trained_at
                    ? dayjs(activeModel.trained_at).format("MMM DD, YYYY")
                    : "N/A"
                }
              />
            </Col>
          </Row>
        </Card>
      )}

      {/* Training Config + Retrain Controls */}
      <Card
        title={
          <Space>
            <SettingOutlined />
            Training Configuration
          </Space>
        }
        style={{ marginBottom: 24 }}
      >
        <Row gutter={[24, 16]}>
          {/* Auto-retrain config */}
          <Col xs={24} md={12}>
            <Space direction="vertical" size="small" style={{ width: "100%" }}>
              <Text strong>Auto-retrain Threshold</Text>
              <Text type="secondary">
                Automatically trigger retraining after collecting this many new detections.
              </Text>
              <Space>
                <InputNumber
                  min={10}
                  max={10000}
                  step={10}
                  value={autoRetrainThreshold}
                  onChange={(v) => v && setAutoRetrainThreshold(v)}
                  addonAfter="detections"
                  style={{ width: 200 }}
                />
                <Button
                  onClick={() => saveAutoRetrain(autoRetrainThreshold)}
                >
                  Save
                </Button>
              </Space>
              <Text type="secondary" style={{ fontSize: 12 }}>
                Current threshold: {autoRetrainThreshold} detections (stored locally)
              </Text>
            </Space>
          </Col>

          <Col xs={0} md={0}>
            <Divider type="vertical" style={{ height: "100%" }} />
          </Col>

          {/* Manual retrain */}
          <Col xs={24} md={12}>
            <Space direction="vertical" size="small" style={{ width: "100%" }}>
              <Text strong>Manual Training</Text>
              <Text type="secondary">
                Run training immediately using all collected training data.
              </Text>
              <Space align="center" size="middle">
                <InputNumber
                  min={1}
                  max={500}
                  value={retrainEpochs}
                  onChange={(v) => setRetrainEpochs(v || 50)}
                  addonBefore="Epochs"
                  style={{ width: 160 }}
                />
                <Button
                  type="primary"
                  icon={retraining ? <SyncOutlined spin /> : <ThunderboltOutlined />}
                  onClick={handleRetrain}
                  loading={retraining}
                >
                  {retraining ? "Training..." : "Train Now"}
                </Button>
              </Space>
            </Space>
          </Col>
        </Row>

        {/* Training progress/result */}
        {retraining && (
          <div style={{ marginTop: 16 }}>
            <Progress
              percent={99}
              status="active"
              strokeColor={{ from: "#108ee9", to: "#87d068" }}
            />
            <Text type="secondary">
              Training in progress... This may take a few minutes depending on data size.
            </Text>
          </div>
        )}

        {retrainResult && !retraining && (
          <div style={{ marginTop: 16 }}>
            {retrainResult.status === "trained" ? (
              <Alert
                type="success"
                showIcon
                message="Training Completed"
                description={
                  <Descriptions column={3} size="small" style={{ marginTop: 8 }}>
                    <Descriptions.Item label="Examples">
                      {retrainResult.num_examples}
                    </Descriptions.Item>
                    <Descriptions.Item label="Train/Val">
                      {retrainResult.train_size} / {retrainResult.val_size}
                    </Descriptions.Item>
                    <Descriptions.Item label="Labels">
                      {retrainResult.num_labels}
                    </Descriptions.Item>
                    <Descriptions.Item label="Val Accuracy">
                      {retrainResult.val_accuracy != null
                        ? `${(retrainResult.val_accuracy * 100).toFixed(1)}%`
                        : "-"}
                    </Descriptions.Item>
                    <Descriptions.Item label="Val F1">
                      {retrainResult.val_weighted_f1 != null
                        ? `${(retrainResult.val_weighted_f1 * 100).toFixed(1)}%`
                        : "-"}
                    </Descriptions.Item>
                    <Descriptions.Item label="Best Epoch">
                      {retrainResult.best_epoch} / {retrainResult.stopped_epoch}
                      {retrainResult.early_stopped && " (early stopped)"}
                    </Descriptions.Item>
                  </Descriptions>
                }
              />
            ) : retrainResult.status === "skipped" ? (
              <Alert
                type="warning"
                showIcon
                message="Training Skipped"
                description={retrainResult.reason}
              />
            ) : (
              <Alert
                type="info"
                showIcon
                message={`Training Status: ${retrainResult.status}`}
              />
            )}
          </div>
        )}
      </Card>

      <Row gutter={16} style={{ marginBottom: 24 }}>
        {/* Loss Curve */}
        <Col xs={24} lg={12}>
          <Card title="Training Loss" size="small">
            {lossChartData.length > 0 ? (
              <Line {...lossChartConfig} />
            ) : (
              <div style={{ textAlign: "center", padding: 40, color: "#999" }}>
                No training history available yet. Train a model to see loss curves.
              </div>
            )}
          </Card>
        </Col>

        {/* Per-label Metrics */}
        <Col xs={24} lg={12}>
          <Card title="Per-Label Metrics" size="small">
            <Table
              dataSource={
                latestPerLabel
                  ? Object.entries(latestPerLabel).map(([label, m]) => ({
                      key: label,
                      label,
                      ...m,
                    }))
                  : []
              }
              columns={[
                {
                  title: "Label",
                  dataIndex: "label",
                  key: "label",
                  render: (l: string) => <Tag>{l}</Tag>,
                },
                {
                  title: "Precision",
                  dataIndex: "precision",
                  key: "precision",
                  render: (v: number) => `${((v || 0) * 100).toFixed(1)}%`,
                },
                {
                  title: "Recall",
                  dataIndex: "recall",
                  key: "recall",
                  render: (v: number) => `${((v || 0) * 100).toFixed(1)}%`,
                },
                {
                  title: "F1",
                  dataIndex: "f1",
                  key: "f1",
                  render: (v: number) => `${((v || 0) * 100).toFixed(1)}%`,
                },
                {
                  title: "Support",
                  dataIndex: "support",
                  key: "support",
                },
              ]}
              size="small"
              pagination={false}
              scroll={{ y: 200 }}
              locale={{
                emptyText: "Train a model to see per-label metrics.",
              }}
            />
          </Card>
        </Col>
      </Row>

      {/* Version History */}
      <Card title="Version History">
        <Table
          columns={versionColumns}
          dataSource={versions}
          rowKey="id"
          size="small"
          pagination={{ pageSize: 10 }}
          locale={{
            emptyText: "No model versions yet. Train your first model above.",
          }}
        />
      </Card>
    </div>
  );
};

export default ModelsPage;
