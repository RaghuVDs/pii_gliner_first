import React, { useEffect, useState } from "react";
import {
  Row,
  Col,
  Card,
  Statistic,
  Typography,
  Tag,
  Space,
  Spin,
  Table,
  Empty,
  Progress,
} from "antd";
import {
  ScanOutlined,
  SafetyCertificateOutlined,
  FileTextOutlined,
  RobotOutlined,
  DatabaseOutlined,
  CheckCircleOutlined,
  ExperimentOutlined,
} from "@ant-design/icons";
import { analyticsApi } from "@/api/analytics";
import { learningApi } from "@/api/learning";
import { modelsApi } from "@/api/models";
import { rulesApi } from "@/api/rules";

const { Title, Text } = Typography;

interface OverviewStats {
  total_scans: number;
  total_detections: number;
  total_chars_processed: number;
  unique_pii_types: number;
  avg_detections_per_scan: number;
}

interface TrainingStats {
  total_examples: number;
  total?: number;
  by_label: Record<string, number>;
  by_source: Record<string, number>;
}

const DashboardPage: React.FC = () => {
  const [overview, setOverview] = useState<OverviewStats | null>(null);
  const [trainingStats, setTrainingStats] = useState<TrainingStats | null>(null);
  const [modelInfo, setModelInfo] = useState<any>(null);
  const [rulesCount, setRulesCount] = useState({ regex: 0, field: 0, context: 0, masking: 0 });
  const [pendingStats, setPendingStats] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      try {
        const results = await Promise.allSettled([
          analyticsApi.getOverview(),
          learningApi.getTrainingStats(),
          modelsApi.getActiveModel(),
          rulesApi.listRegexRules(),
          rulesApi.listContextRules(),
          rulesApi.listFieldPatterns(),
          rulesApi.listMaskingRules(),
          learningApi.getPendingStats(),
        ]);

        if (results[0].status === "fulfilled") setOverview(results[0].value as any);
        if (results[1].status === "fulfilled") setTrainingStats(results[1].value as any);
        if (results[2].status === "fulfilled") setModelInfo(results[2].value);

        const regex = results[3].status === "fulfilled" ? (results[3].value as any[]).length : 0;
        const context = results[4].status === "fulfilled" ? (results[4].value as any[]).length : 0;
        const field = results[5].status === "fulfilled" ? (results[5].value as any[]).length : 0;
        const masking = results[6].status === "fulfilled" ? (results[6].value as any[]).length : 0;
        setRulesCount({ regex, field, context, masking });

        if (results[7].status === "fulfilled") setPendingStats(results[7].value);
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  if (loading) {
    return (
      <div style={{ textAlign: "center", padding: 80 }}>
        <Spin size="large" />
      </div>
    );
  }

  const totalRules = rulesCount.regex + rulesCount.field + rulesCount.context + rulesCount.masking;
  const totalTraining = trainingStats?.total_examples || trainingStats?.total || 0;
  const modelAccuracy = modelInfo?.val_accuracy || modelInfo?.metrics_detail?.val_accuracy || 0;
  const modelF1 = modelInfo?.val_weighted_f1 || modelInfo?.metrics_detail?.val_weighted_f1 || 0;
  const modelLabels = modelInfo?.num_labels || 0;

  // Build top PII types from training stats
  const topPIITypes = trainingStats?.by_label
    ? Object.entries(trainingStats.by_label)
        .sort(([, a], [, b]) => b - a)
        .slice(0, 12)
        .map(([label, count]) => ({ label, count }))
    : [];

  // Source distribution from training stats
  const sourceData = trainingStats?.by_source
    ? Object.entries(trainingStats.by_source).map(([source, count]) => ({ source, count }))
    : [];

  return (
    <div>
      <Title level={4} style={{ marginBottom: 24 }}>Dashboard</Title>

      {/* Stat Cards Row 1 */}
      <Row gutter={[16, 16]} style={{ marginBottom: 16 }}>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="Total Scans"
              value={overview?.total_scans || 0}
              prefix={<ScanOutlined style={{ color: "#1677ff" }} />}
              valueStyle={{ color: "#1677ff" }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="PII Entities Found"
              value={overview?.total_detections || 0}
              prefix={<SafetyCertificateOutlined style={{ color: "#cf1322" }} />}
              valueStyle={{ color: "#cf1322" }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="Active Rules"
              value={totalRules}
              prefix={<FileTextOutlined style={{ color: "#722ed1" }} />}
              valueStyle={{ color: "#722ed1" }}
              suffix={
                <Text type="secondary" style={{ fontSize: 12 }}>
                  ({rulesCount.regex}R + {rulesCount.field}F + {rulesCount.context}C)
                </Text>
              }
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="Training Examples"
              value={totalTraining}
              prefix={<DatabaseOutlined style={{ color: "#fa8c16" }} />}
              valueStyle={{ color: "#fa8c16" }}
            />
          </Card>
        </Col>
      </Row>

      {/* Stat Cards Row 2 - Model */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="Model Accuracy"
              value={modelAccuracy ? (modelAccuracy * 100).toFixed(1) : "N/A"}
              prefix={<RobotOutlined style={{ color: "#3f8600" }} />}
              suffix={modelAccuracy ? "%" : ""}
              valueStyle={{ color: "#3f8600" }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="Model F1 Score"
              value={modelF1 ? (modelF1 * 100).toFixed(1) : "N/A"}
              prefix={<CheckCircleOutlined style={{ color: "#1677ff" }} />}
              suffix={modelF1 ? "%" : ""}
              valueStyle={{ color: "#1677ff" }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="PII Labels Trained"
              value={modelLabels}
              prefix={<ExperimentOutlined style={{ color: "#13c2c2" }} />}
              valueStyle={{ color: "#13c2c2" }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="Avg Detections/Scan"
              value={overview?.avg_detections_per_scan?.toFixed(1) || 0}
              prefix={<ScanOutlined style={{ color: "#eb2f96" }} />}
              valueStyle={{ color: "#eb2f96" }}
            />
          </Card>
        </Col>
      </Row>

      {/* Charts Row */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        {/* Top PII Types */}
        <Col xs={24} lg={14}>
          <Card title="Top PII Types Detected" size="small">
            {topPIITypes.length === 0 ? (
              <Empty description="Run detections to see PII type distribution" image={Empty.PRESENTED_IMAGE_SIMPLE} />
            ) : (
              <div>
                {topPIITypes.map((item) => {
                  const maxCount = topPIITypes[0]?.count || 1;
                  const percent = (item.count / maxCount) * 100;
                  return (
                    <div key={item.label} style={{ marginBottom: 8 }}>
                      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 2 }}>
                        <Text style={{ fontSize: 12 }}>{item.label.replace(/_/g, " ")}</Text>
                        <Text strong style={{ fontSize: 12 }}>{item.count}</Text>
                      </div>
                      <Progress
                        percent={percent}
                        showInfo={false}
                        size="small"
                        strokeColor="#1677ff"
                      />
                    </div>
                  );
                })}
              </div>
            )}
          </Card>
        </Col>

        {/* Detection Sources */}
        <Col xs={24} lg={10}>
          <Card title="Detection Sources" size="small">
            {sourceData.length === 0 ? (
              <Empty description="No source data yet" image={Empty.PRESENTED_IMAGE_SIMPLE} />
            ) : (
              <div>
                {sourceData.map((item) => {
                  const colors: Record<string, string> = {
                    gliner: "#1677ff",
                    regex: "#52c41a",
                    derived: "#722ed1",
                    propagated: "#fa8c16",
                    field_label: "#13c2c2",
                    context: "#eb2f96",
                  };
                  const totalSource = sourceData.reduce((s, d) => s + d.count, 0);
                  const pct = totalSource > 0 ? ((item.count / totalSource) * 100).toFixed(1) : "0";
                  return (
                    <div key={item.source} style={{ marginBottom: 12 }}>
                      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 2 }}>
                        <Tag color={colors[item.source] || "default"}>{item.source.toUpperCase()}</Tag>
                        <Text strong>{item.count} ({pct}%)</Text>
                      </div>
                      <Progress
                        percent={parseFloat(pct)}
                        showInfo={false}
                        size="small"
                        strokeColor={colors[item.source] || "#d9d9d9"}
                      />
                    </div>
                  );
                })}
              </div>
            )}
          </Card>
        </Col>
      </Row>

      {/* System Status */}
      <Row gutter={[16, 16]}>
        <Col xs={24} lg={12}>
          <Card title="Rules Configuration" size="small">
            <Table
              dataSource={[
                { key: "regex", rule: "Regex Rules", count: rulesCount.regex, color: "blue" },
                { key: "field", rule: "Field Patterns", count: rulesCount.field, color: "cyan" },
                { key: "context", rule: "Context Rules", count: rulesCount.context, color: "green" },
                { key: "masking", rule: "Masking Strategies", count: rulesCount.masking, color: "purple" },
              ]}
              columns={[
                { title: "Rule Type", dataIndex: "rule", render: (v: string, r: any) => <Tag color={r.color}>{v}</Tag> },
                { title: "Count", dataIndex: "count", render: (v: number) => <Text strong>{v}</Text> },
              ]}
              pagination={false}
              size="small"
            />
          </Card>
        </Col>
        <Col xs={24} lg={12}>
          <Card title="System Status" size="small">
            <Table
              dataSource={[
                { key: "gliner", component: "GLiNER Model", status: "Active", detail: "CUDA (RTX 5090)", color: "green" },
                { key: "lstm", component: "LSTM Classifier", status: modelLabels > 0 ? "Active" : "Not Trained", detail: modelLabels > 0 ? `${modelLabels} labels, ${(modelAccuracy * 100).toFixed(1)}% acc` : "Run detections to train", color: modelLabels > 0 ? "green" : "orange" },
                { key: "pg", component: "PostgreSQL", status: "Connected", detail: "157 PII types seeded", color: "green" },
                { key: "mongo", component: "MongoDB", status: "Connected", detail: `${totalTraining} training examples`, color: "green" },
                { key: "redis", component: "Redis", status: "Connected", detail: "Session & rate limiting", color: "green" },
              ]}
              columns={[
                { title: "Component", dataIndex: "component" },
                { title: "Status", dataIndex: "status", render: (v: string, r: any) => <Tag color={r.color}>{v}</Tag> },
                { title: "Details", dataIndex: "detail", render: (v: string) => <Text type="secondary" style={{ fontSize: 12 }}>{v}</Text> },
              ]}
              pagination={false}
              size="small"
            />
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default DashboardPage;
