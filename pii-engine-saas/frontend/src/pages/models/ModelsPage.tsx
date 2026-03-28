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
} from "antd";
import {
  RobotOutlined,
  ThunderboltOutlined,
  CheckCircleOutlined,
  SyncOutlined,
} from "@ant-design/icons";
import { Line } from "@ant-design/charts";
import { modelsApi } from "@/api/models";
import { ModelVersion, ModelMetrics } from "@/types/models";
import dayjs from "dayjs";

const { Title, Text } = Typography;

const ModelsPage: React.FC = () => {
  const [activeModel, setActiveModel] = useState<ModelVersion | null>(null);
  const [versions, setVersions] = useState<ModelVersion[]>([]);
  const [metrics, setMetrics] = useState<ModelMetrics | null>(null);
  const [loading, setLoading] = useState(true);
  const [retrainEpochs, setRetrainEpochs] = useState<number>(10);
  const [retraining, setRetraining] = useState(false);
  const [retrainJobId, setRetrainJobId] = useState<string | null>(null);
  const [retrainProgress, setRetrainProgress] = useState(0);

  const fetchData = useCallback(async () => {
    setLoading(true);
    try {
      const [activeData, versionsData, metricsData] = await Promise.all([
        modelsApi.getActiveModel(),
        modelsApi.listVersions(),
        modelsApi.getMetrics(),
      ]);
      setActiveModel(activeData);
      setVersions(versionsData);
      setMetrics(metricsData);
    } catch {
      message.error("Failed to load model data");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  // Poll retrain status
  useEffect(() => {
    if (!retrainJobId) return;
    const interval = setInterval(async () => {
      try {
        const status = await modelsApi.getRetrainStatus(retrainJobId);
        setRetrainProgress(status.progress);
        if (status.status === "completed") {
          setRetraining(false);
          setRetrainJobId(null);
          message.success("Model retraining completed!");
          fetchData();
        } else if (status.status === "failed") {
          setRetraining(false);
          setRetrainJobId(null);
          message.error(`Retraining failed: ${status.message}`);
        }
      } catch {
        // continue polling
      }
    }, 3000);
    return () => clearInterval(interval);
  }, [retrainJobId, fetchData]);

  const handleRetrain = async () => {
    setRetraining(true);
    try {
      const result = await modelsApi.triggerRetrain(retrainEpochs);
      setRetrainJobId(result.job_id);
      setRetrainProgress(0);
      message.info("Model retraining started");
    } catch {
      setRetraining(false);
      message.error("Failed to start retraining");
    }
  };

  const handleActivate = async (versionId: string) => {
    Modal.confirm({
      title: "Activate Model Version",
      content: "This will replace the currently active model. Continue?",
      onOk: async () => {
        try {
          await modelsApi.activateVersion(versionId);
          message.success("Model version activated");
          fetchData();
        } catch {
          message.error("Failed to activate model version");
        }
      },
    });
  };

  const lossChartConfig = {
    data: (metrics?.loss_history || []).map((loss, epoch) => ({
      epoch: epoch + 1,
      loss,
    })),
    xField: "epoch",
    yField: "loss",
    smooth: true,
    height: 250,
    color: "#ff4d4f",
  };

  const versionColumns = [
    {
      title: "Version",
      dataIndex: "version",
      key: "version",
      render: (version: string, record: ModelVersion) => (
        <Space>
          <Text strong>{version}</Text>
          {record.is_active && <Tag color="green">ACTIVE</Tag>}
        </Space>
      ),
    },
    {
      title: "Accuracy",
      dataIndex: "accuracy",
      key: "accuracy",
      render: (v: number) => `${(v * 100).toFixed(1)}%`,
    },
    {
      title: "F1 Score",
      dataIndex: "f1_score",
      key: "f1_score",
      render: (v: number) => `${(v * 100).toFixed(1)}%`,
    },
    {
      title: "Labels",
      dataIndex: "labels_count",
      key: "labels_count",
    },
    {
      title: "Training Data",
      dataIndex: "training_examples_count",
      key: "training_examples_count",
    },
    {
      title: "Epochs",
      dataIndex: "epochs",
      key: "epochs",
    },
    {
      title: "Trained",
      dataIndex: "trained_at",
      key: "trained_at",
      render: (date: string) => dayjs(date).format("MMM DD, YYYY HH:mm"),
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
              <Tag color="green">{activeModel.version}</Tag>
            </Space>
          }
          style={{ marginBottom: 24 }}
        >
          <Row gutter={24}>
            <Col span={6}>
              <Statistic
                title="Accuracy"
                value={(activeModel.accuracy * 100).toFixed(1)}
                suffix="%"
                prefix={<CheckCircleOutlined />}
                valueStyle={{ color: "#3f8600" }}
              />
            </Col>
            <Col span={6}>
              <Statistic
                title="F1 Score"
                value={(activeModel.f1_score * 100).toFixed(1)}
                suffix="%"
              />
            </Col>
            <Col span={6}>
              <Statistic
                title="Labels"
                value={activeModel.labels_count}
              />
            </Col>
            <Col span={6}>
              <Statistic
                title="Trained"
                value={dayjs(activeModel.trained_at).format("MMM DD, YYYY")}
              />
            </Col>
          </Row>
        </Card>
      )}

      {/* Retrain Controls */}
      <Card style={{ marginBottom: 24 }}>
        <Space align="center" size="large">
          <Text strong>Retrain Model</Text>
          <InputNumber
            min={1}
            max={100}
            value={retrainEpochs}
            onChange={(v) => setRetrainEpochs(v || 10)}
            addonBefore="Epochs"
          />
          <Button
            type="primary"
            icon={retraining ? <SyncOutlined spin /> : <ThunderboltOutlined />}
            onClick={handleRetrain}
            loading={retraining}
          >
            {retraining ? "Retraining..." : "Start Retrain"}
          </Button>
          {retraining && (
            <Progress
              percent={Math.round(retrainProgress * 100)}
              style={{ width: 200 }}
            />
          )}
        </Space>
      </Card>

      <Row gutter={16} style={{ marginBottom: 24 }}>
        {/* Loss Curve */}
        <Col xs={24} lg={12}>
          <Card title="Training Loss" size="small">
            <Line {...lossChartConfig} />
          </Card>
        </Col>

        {/* Per-label Metrics */}
        <Col xs={24} lg={12}>
          <Card title="Per-Label Metrics" size="small">
            <Table
              dataSource={
                metrics
                  ? Object.entries(metrics.per_label).map(
                      ([label, m]) => ({
                        key: label,
                        label,
                        ...m,
                      })
                    )
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
                  render: (v: number) => `${(v * 100).toFixed(1)}%`,
                },
                {
                  title: "Recall",
                  dataIndex: "recall",
                  key: "recall",
                  render: (v: number) => `${(v * 100).toFixed(1)}%`,
                },
                {
                  title: "F1",
                  dataIndex: "f1_score",
                  key: "f1_score",
                  render: (v: number) => `${(v * 100).toFixed(1)}%`,
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
        />
      </Card>
    </div>
  );
};

export default ModelsPage;
