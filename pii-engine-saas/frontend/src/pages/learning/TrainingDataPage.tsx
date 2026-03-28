import React, { useEffect, useState, useCallback } from "react";
import {
  Card,
  Table,
  Select,
  Typography,
  Tag,
  Space,
  Popconfirm,
  Button,
  Row,
  Col,
  message,
} from "antd";
import { DeleteOutlined } from "@ant-design/icons";
import { Bar } from "@ant-design/charts";
import { learningApi, TrainingStats } from "@/api/learning";
import { TrainingExample } from "@/types/models";
import { DetectionSource } from "@/types/enums";
import { usePagination } from "@/hooks/usePagination";
import SourceBadge from "@/components/common/SourceBadge";
import ScoreIndicator from "@/components/common/ScoreIndicator";
import dayjs from "dayjs";

const { Title } = Typography;

const TrainingDataPage: React.FC = () => {
  const [data, setData] = useState<TrainingExample[]>([]);
  const [stats, setStats] = useState<TrainingStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [filterLabel, setFilterLabel] = useState<string | undefined>();
  const { page, pageSize, setTotal, paginationProps } = usePagination();

  const fetchData = useCallback(async () => {
    setLoading(true);
    try {
      const [examplesData, statsData] = await Promise.all([
        learningApi.listTrainingData(page, pageSize, filterLabel),
        learningApi.getTrainingStats(),
      ]);
      setData(examplesData.items);
      setTotal(examplesData.total);
      setStats(statsData);
    } catch {
      message.error("Failed to load training data");
    } finally {
      setLoading(false);
    }
  }, [page, pageSize, filterLabel, setTotal]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  const handleDelete = async (id: string) => {
    try {
      await learningApi.deleteTrainingExample(id);
      message.success("Training example deleted");
      fetchData();
    } catch {
      message.error("Failed to delete training example");
    }
  };

  const labelChartData = stats
    ? Object.entries(stats.by_label).map(([label, count]) => ({
        label,
        count,
      }))
    : [];

  const labelChartConfig = {
    data: labelChartData,
    xField: "label",
    yField: "count",
    height: 250,
    color: "#1677ff",
  };

  const labels = stats ? Object.keys(stats.by_label) : [];

  const columns = [
    {
      title: "Structure",
      dataIndex: "structure",
      key: "structure",
      ellipsis: true,
      render: (text: string) => (
        <code style={{ fontSize: 12 }}>{text}</code>
      ),
    },
    {
      title: "Label",
      dataIndex: "label",
      key: "label",
      render: (label: string) => <Tag color="blue">{label}</Tag>,
    },
    {
      title: "Source",
      dataIndex: "source",
      key: "source",
      render: (source: DetectionSource) => <SourceBadge source={source} />,
    },
    {
      title: "Score",
      dataIndex: "score",
      key: "score",
      width: 120,
      render: (score: number) => (
        <ScoreIndicator score={score} size="small" showLabel={false} />
      ),
    },
    {
      title: "Created",
      dataIndex: "created_at",
      key: "created_at",
      width: 140,
      render: (date: string) => dayjs(date).format("MMM DD, YYYY"),
    },
    {
      title: "Actions",
      key: "actions",
      width: 80,
      render: (_: unknown, record: TrainingExample) => (
        <Popconfirm
          title="Delete this example?"
          onConfirm={() => handleDelete(record.id)}
        >
          <Button
            type="text"
            icon={<DeleteOutlined />}
            size="small"
            danger
          />
        </Popconfirm>
      ),
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
          Training Data
        </Title>
        <Space>
          <Select
            placeholder="Filter by label"
            style={{ width: 200 }}
            value={filterLabel}
            onChange={setFilterLabel}
            allowClear
            options={labels.map((l) => ({ label: l, value: l }))}
          />
        </Space>
      </div>

      {/* Stats */}
      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col xs={24} lg={12}>
          <Card title="Label Distribution" size="small">
            <Bar {...labelChartConfig} />
          </Card>
        </Col>
        <Col xs={24} lg={12}>
          <Card title="Summary" size="small">
            <Space direction="vertical" size="middle">
              <div>
                <strong>Total Examples:</strong> {stats?.total || 0}
              </div>
              <div>
                <strong>Unique Labels:</strong>{" "}
                {stats ? Object.keys(stats.by_label).length : 0}
              </div>
              <div>
                <strong>By Source:</strong>
                <div style={{ marginTop: 8 }}>
                  {stats &&
                    Object.entries(stats.by_source).map(
                      ([source, count]) => (
                        <Tag key={source} style={{ margin: 4 }}>
                          {source}: {count}
                        </Tag>
                      )
                    )}
                </div>
              </div>
            </Space>
          </Card>
        </Col>
      </Row>

      {/* Data Table */}
      <Card>
        <Table
          columns={columns}
          dataSource={data}
          rowKey="id"
          loading={loading}
          size="small"
          pagination={paginationProps}
          scroll={{ x: 800 }}
        />
      </Card>
    </div>
  );
};

export default TrainingDataPage;
