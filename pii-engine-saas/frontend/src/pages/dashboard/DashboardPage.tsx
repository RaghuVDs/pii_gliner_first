import React, { useEffect, useState } from "react";
import {
  Row,
  Col,
  Card,
  Statistic,
  Typography,
  Table,
  Tag,
  Alert,
  Space,
  Spin,
} from "antd";
import {
  ScanOutlined,
  SafetyCertificateOutlined,
  FileTextOutlined,
  RobotOutlined,
  ArrowUpOutlined,
} from "@ant-design/icons";
import { Line, Bar, Pie } from "@ant-design/charts";
import { analyticsApi, OverviewData, TimelinePoint, TypeBreakdown, SourceBreakdown } from "@/api/analytics";
import { detectionApi } from "@/api/detection";
import { learningApi } from "@/api/learning";
import { DetectionJob } from "@/types/models";
import dayjs from "dayjs";

const { Title } = Typography;

const DashboardPage: React.FC = () => {
  const [overview, setOverview] = useState<OverviewData | null>(null);
  const [timeline, setTimeline] = useState<TimelinePoint[]>([]);
  const [topTypes, setTopTypes] = useState<TypeBreakdown[]>([]);
  const [sourceBreakdown, setSourceBreakdown] = useState<SourceBreakdown[]>([]);
  const [recentJobs, setRecentJobs] = useState<DetectionJob[]>([]);
  const [pendingCount, setPendingCount] = useState(0);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      try {
        const [overviewData, timelineData, topTypesData, sourceData, jobsData, pendingStats] =
          await Promise.allSettled([
            analyticsApi.getOverview(),
            analyticsApi.getTimeline(),
            analyticsApi.getTopTypes(10),
            analyticsApi.getBySource(),
            detectionApi.listJobs(1, 5),
            learningApi.getPendingStats(),
          ]);

        if (overviewData.status === "fulfilled") setOverview(overviewData.value);
        if (timelineData.status === "fulfilled") setTimeline(timelineData.value);
        if (topTypesData.status === "fulfilled") setTopTypes(topTypesData.value);
        if (sourceData.status === "fulfilled") setSourceBreakdown(sourceData.value);
        if (jobsData.status === "fulfilled") setRecentJobs(jobsData.value.items);
        if (pendingStats.status === "fulfilled") setPendingCount(pendingStats.value.pending);
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

  const timelineConfig = {
    data: timeline,
    xField: "timestamp",
    yField: "detections",
    smooth: true,
    height: 300,
    xAxis: {
      label: {
        formatter: (v: string) => dayjs(v).format("MMM DD"),
      },
    },
    color: "#1677ff",
  };

  const barConfig = {
    data: topTypes,
    xField: "pii_type",
    yField: "count",
    height: 300,
    color: "#1677ff",
    xAxis: {
      label: {
        autoRotate: true,
        autoHide: false,
      },
    },
  };

  const pieConfig = {
    data: sourceBreakdown,
    angleField: "count",
    colorField: "source",
    height: 300,
    radius: 0.8,
    innerRadius: 0.6,
    label: {
      text: "source",
      position: "outside" as const,
    },
    legend: {
      position: "bottom" as const,
    },
  };

  const jobColumns = [
    {
      title: "Job ID",
      dataIndex: "id",
      key: "id",
      render: (id: string) => id.slice(0, 8) + "...",
    },
    {
      title: "Status",
      dataIndex: "status",
      key: "status",
      render: (status: string) => {
        const colorMap: Record<string, string> = {
          completed: "green",
          processing: "blue",
          pending: "orange",
          failed: "red",
          cancelled: "default",
        };
        return <Tag color={colorMap[status] || "default"}>{status.toUpperCase()}</Tag>;
      },
    },
    {
      title: "Results",
      dataIndex: "result_count",
      key: "result_count",
    },
    {
      title: "Created",
      dataIndex: "created_at",
      key: "created_at",
      render: (date: string) => dayjs(date).format("MMM DD, HH:mm"),
    },
  ];

  return (
    <div>
      <Title level={4} style={{ marginBottom: 24 }}>
        Dashboard
      </Title>

      {/* Alerts */}
      {pendingCount > 0 && (
        <Alert
          message={`${pendingCount} pending rules ready for review`}
          type="info"
          showIcon
          closable
          style={{ marginBottom: 16 }}
          action={
            <a href="/learning/pending-rules">Review Now</a>
          }
        />
      )}

      {/* Stat Cards */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="Total Scans"
              value={overview?.total_scans || 0}
              prefix={<ScanOutlined />}
              suffix={
                <span style={{ fontSize: 14, color: "#52c41a" }}>
                  <ArrowUpOutlined /> {overview?.scans_today || 0} today
                </span>
              }
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="PII Found"
              value={overview?.total_pii_found || 0}
              prefix={<SafetyCertificateOutlined />}
              valueStyle={{ color: "#cf1322" }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="Active Rules"
              value={overview?.active_rules || 0}
              prefix={<FileTextOutlined />}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="Model Accuracy"
              value={overview?.model_accuracy ? (overview.model_accuracy * 100).toFixed(1) : 0}
              prefix={<RobotOutlined />}
              suffix="%"
              valueStyle={{ color: "#3f8600" }}
            />
          </Card>
        </Col>
      </Row>

      {/* Charts */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col xs={24} lg={16}>
          <Card title="Detection Timeline">
            <Line {...timelineConfig} />
          </Card>
        </Col>
        <Col xs={24} lg={8}>
          <Card title="Detection Sources">
            <Pie {...pieConfig} />
          </Card>
        </Col>
      </Row>

      <Row gutter={[16, 16]}>
        <Col xs={24} lg={14}>
          <Card title="Top PII Types">
            <Bar {...barConfig} />
          </Card>
        </Col>
        <Col xs={24} lg={10}>
          <Card title="Recent Jobs">
            <Table
              columns={jobColumns}
              dataSource={recentJobs}
              rowKey="id"
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
