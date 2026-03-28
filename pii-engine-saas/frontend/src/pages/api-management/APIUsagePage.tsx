import React, { useEffect, useState } from "react";
import {
  Card,
  Typography,
  Row,
  Col,
  Statistic,
  Table,
  DatePicker,
  Space,
  Tag,
  message,
  Spin,
} from "antd";
import {
  ApiOutlined,
  WarningOutlined,
  ClockCircleOutlined,
} from "@ant-design/icons";
import { Line } from "@ant-design/charts";
import { apiKeysApi, UsageAnalytics } from "@/api/apiKeys";
import dayjs from "dayjs";

const { Title, Text } = Typography;
const { RangePicker } = DatePicker;

const APIUsagePage: React.FC = () => {
  const [analytics, setAnalytics] = useState<UsageAnalytics | null>(null);
  const [loading, setLoading] = useState(true);
  const [dateRange, setDateRange] = useState<
    [dayjs.Dayjs, dayjs.Dayjs] | null
  >(null);

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      try {
        const data = await apiKeysApi.getUsageAnalytics(
          dateRange?.[0]?.format("YYYY-MM-DD"),
          dateRange?.[1]?.format("YYYY-MM-DD")
        );
        setAnalytics(data);
      } catch {
        message.error("Failed to load usage analytics");
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, [dateRange]);

  if (loading) {
    return (
      <div style={{ textAlign: "center", padding: 80 }}>
        <Spin size="large" />
      </div>
    );
  }

  const timelineConfig = {
    data: analytics?.requests_by_day || [],
    xField: "date",
    yField: "count",
    smooth: true,
    height: 300,
    color: "#1677ff",
    xAxis: {
      label: {
        formatter: (v: string) => dayjs(v).format("MMM DD"),
      },
    },
  };

  const endpointData = analytics?.requests_by_endpoint
    ? Object.entries(analytics.requests_by_endpoint).map(
        ([endpoint, count]) => ({
          endpoint,
          count,
          key: endpoint,
        })
      )
    : [];

  const endpointColumns = [
    {
      title: "Endpoint",
      dataIndex: "endpoint",
      key: "endpoint",
      render: (endpoint: string) => (
        <Text code>{endpoint}</Text>
      ),
    },
    {
      title: "Requests",
      dataIndex: "count",
      key: "count",
      sorter: (
        a: { count: number },
        b: { count: number }
      ) => a.count - b.count,
    },
    {
      title: "% of Total",
      key: "percentage",
      render: (_: unknown, record: { count: number }) => {
        const pct = analytics
          ? ((record.count / analytics.total_requests) * 100).toFixed(1)
          : "0";
        return <Tag>{pct}%</Tag>;
      },
    },
  ];

  return (
    <div>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: 24,
        }}
      >
        <Title level={4} style={{ margin: 0 }}>
          API Usage Analytics
        </Title>
        <RangePicker
          onChange={(dates) =>
            setDateRange(
              dates as [dayjs.Dayjs, dayjs.Dayjs] | null
            )
          }
        />
      </div>

      {/* Stats */}
      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col xs={24} sm={8}>
          <Card>
            <Statistic
              title="Total Requests"
              value={analytics?.total_requests || 0}
              prefix={<ApiOutlined />}
            />
          </Card>
        </Col>
        <Col xs={24} sm={8}>
          <Card>
            <Statistic
              title="Total Errors"
              value={analytics?.total_errors || 0}
              prefix={<WarningOutlined />}
              valueStyle={{
                color:
                  (analytics?.total_errors || 0) > 0 ? "#cf1322" : undefined,
              }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={8}>
          <Card>
            <Statistic
              title="Avg Response Time"
              value={analytics?.avg_response_time_ms?.toFixed(0) || 0}
              suffix="ms"
              prefix={<ClockCircleOutlined />}
            />
          </Card>
        </Col>
      </Row>

      {/* Usage Timeline */}
      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col xs={24} lg={16}>
          <Card title="Request Timeline">
            <Line {...timelineConfig} />
          </Card>
        </Col>
        <Col xs={24} lg={8}>
          <Card title="Rate Limit Status">
            <Space direction="vertical" style={{ width: "100%" }}>
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                }}
              >
                <Text>Current window</Text>
                <Tag color="green">Within Limits</Tag>
              </div>
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                }}
              >
                <Text>Error rate</Text>
                <Text>
                  {analytics && analytics.total_requests > 0
                    ? (
                        (analytics.total_errors / analytics.total_requests) *
                        100
                      ).toFixed(2)
                    : "0"}
                  %
                </Text>
              </div>
            </Space>
          </Card>
        </Col>
      </Row>

      {/* Endpoint Breakdown */}
      <Card title="Endpoint Breakdown">
        <Table
          columns={endpointColumns}
          dataSource={endpointData}
          size="small"
          pagination={false}
        />
      </Card>
    </div>
  );
};

export default APIUsagePage;
