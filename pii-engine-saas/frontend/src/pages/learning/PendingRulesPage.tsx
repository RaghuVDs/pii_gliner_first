import React, { useEffect, useState, useCallback } from "react";
import {
  Card,
  Button,
  Space,
  Typography,
  Tag,
  InputNumber,
  Row,
  Col,
  Select,
  Popconfirm,
  Statistic,
  message,
  Spin,
  Empty,
} from "antd";
import {
  CheckOutlined,
  CloseOutlined,
  ThunderboltOutlined,
  ExperimentOutlined,
} from "@ant-design/icons";
import { learningApi, PendingRuleStats } from "@/api/learning";
import { PendingRule } from "@/types/models";
import { usePagination } from "@/hooks/usePagination";
import ScoreIndicator from "@/components/common/ScoreIndicator";

const { Title, Text, Paragraph } = Typography;

const PendingRulesPage: React.FC = () => {
  const [rules, setRules] = useState<PendingRule[]>([]);
  const [stats, setStats] = useState<PendingRuleStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [autoThreshold, setAutoThreshold] = useState<number>(5);
  const [promotingId, setPromotingId] = useState<string | null>(null);
  const { page, pageSize, setTotal, paginationProps } = usePagination();

  const fetchData = useCallback(async () => {
    setLoading(true);
    try {
      const [rulesData, statsData] = await Promise.all([
        learningApi.listPendingRules(page, pageSize),
        learningApi.getPendingStats(),
      ]);
      setRules(rulesData.items);
      setTotal(rulesData.total);
      setStats(statsData);
    } catch {
      message.error("Failed to load pending rules");
    } finally {
      setLoading(false);
    }
  }, [page, pageSize, setTotal]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  const handlePromote = async (
    ruleId: string,
    label: string,
    keywords?: string[]
  ) => {
    setPromotingId(ruleId);
    try {
      await learningApi.promoteRule(ruleId, label, keywords);
      message.success("Rule promoted successfully");
      fetchData();
    } catch {
      message.error("Failed to promote rule");
    } finally {
      setPromotingId(null);
    }
  };

  const handleReject = async (ruleId: string) => {
    try {
      await learningApi.rejectRule(ruleId);
      message.success("Rule rejected");
      fetchData();
    } catch {
      message.error("Failed to reject rule");
    }
  };

  const handleAutoPromote = async () => {
    try {
      const result = await learningApi.autoPromote(autoThreshold);
      message.success(
        `Auto-promoted ${result.promoted_count} rules`
      );
      fetchData();
    } catch {
      message.error("Auto-promote failed");
    }
  };

  if (loading) {
    return (
      <div style={{ textAlign: "center", padding: 80 }}>
        <Spin size="large" />
      </div>
    );
  }

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
          Pending Rules Review
        </Title>
        <Space>
          <InputNumber
            min={1}
            max={100}
            value={autoThreshold}
            onChange={(v) => setAutoThreshold(v || 5)}
            addonBefore="Threshold"
          />
          <Button
            icon={<ThunderboltOutlined />}
            onClick={handleAutoPromote}
          >
            Auto-Promote
          </Button>
        </Space>
      </div>

      {/* Stats */}
      {stats && (
        <Row gutter={16} style={{ marginBottom: 24 }}>
          <Col span={6}>
            <Card>
              <Statistic title="Total" value={stats.total} />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="Pending"
                value={stats.pending}
                valueStyle={{ color: "#faad14" }}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="Promoted"
                value={stats.promoted}
                valueStyle={{ color: "#52c41a" }}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="Rejected"
                value={stats.rejected}
                valueStyle={{ color: "#ff4d4f" }}
              />
            </Card>
          </Col>
        </Row>
      )}

      {/* Rule Cards */}
      {rules.length === 0 ? (
        <Card>
          <Empty description="No pending rules to review" />
        </Card>
      ) : (
        <Row gutter={[16, 16]}>
          {rules.map((rule) => (
            <Col key={rule.id} xs={24} md={12} xl={8}>
              <Card
                size="small"
                title={
                  <Space>
                    <ExperimentOutlined />
                    <Text strong>{rule.group_key}</Text>
                  </Space>
                }
                extra={
                  <Tag color="blue">
                    Seen {rule.seen_count}x
                  </Tag>
                }
                actions={[
                  <PromoteAction
                    key="promote"
                    rule={rule}
                    loading={promotingId === rule.id}
                    onPromote={handlePromote}
                  />,
                  <Popconfirm
                    key="reject"
                    title="Reject this rule?"
                    onConfirm={() => handleReject(rule.id)}
                  >
                    <Button
                      type="text"
                      danger
                      icon={<CloseOutlined />}
                    >
                      Reject
                    </Button>
                  </Popconfirm>,
                ]}
              >
                <Space direction="vertical" style={{ width: "100%" }}>
                  <div>
                    <Text type="secondary">Suggested Label: </Text>
                    <Tag color="green">{rule.suggested_label}</Tag>
                  </div>
                  <div>
                    <Text type="secondary">Avg Score: </Text>
                    <ScoreIndicator
                      score={rule.avg_score}
                      size="small"
                      showLabel={false}
                    />
                  </div>
                  {rule.keywords.length > 0 && (
                    <div>
                      <Text type="secondary">Keywords: </Text>
                      <Space size={4} wrap>
                        {rule.keywords.map((kw, i) => (
                          <Tag key={i} size="small" >{kw}</Tag>
                        ))}
                      </Space>
                    </div>
                  )}
                  {rule.example_contexts.length > 0 && (
                    <div>
                      <Text type="secondary">Examples:</Text>
                      {rule.example_contexts.slice(0, 3).map((ctx, i) => (
                        <Paragraph
                          key={i}
                          ellipsis={{ rows: 2 }}
                          style={{
                            fontSize: 12,
                            background: "#fafafa",
                            padding: 4,
                            borderRadius: 4,
                            margin: "4px 0",
                          }}
                        >
                          {ctx}
                        </Paragraph>
                      ))}
                    </div>
                  )}
                </Space>
              </Card>
            </Col>
          ))}
        </Row>
      )}

      {/* Pagination */}
      <div style={{ textAlign: "center", marginTop: 24 }}>
        <Space>
          <Button
            disabled={paginationProps.current <= 1}
            onClick={() =>
              paginationProps.onChange(paginationProps.current - 1, pageSize)
            }
          >
            Previous
          </Button>
          <Text>
            Page {paginationProps.current} of{" "}
            {Math.ceil(paginationProps.total / pageSize) || 1}
          </Text>
          <Button
            disabled={
              paginationProps.current >=
              Math.ceil(paginationProps.total / pageSize)
            }
            onClick={() =>
              paginationProps.onChange(paginationProps.current + 1, pageSize)
            }
          >
            Next
          </Button>
        </Space>
      </div>
    </div>
  );
};

// Promote action with label picker
const PromoteAction: React.FC<{
  rule: PendingRule;
  loading: boolean;
  onPromote: (ruleId: string, label: string, keywords?: string[]) => void;
}> = ({ rule, loading, onPromote }) => {
  const [label, setLabel] = useState(rule.suggested_label);

  return (
    <Space>
      <Select
        size="small"
        value={label}
        onChange={setLabel}
        style={{ width: 120 }}
        options={[
          { label: rule.suggested_label, value: rule.suggested_label },
          { label: "PERSON", value: "PERSON" },
          { label: "EMAIL", value: "EMAIL" },
          { label: "PHONE_NUMBER", value: "PHONE_NUMBER" },
          { label: "ADDRESS", value: "ADDRESS" },
          { label: "SSN", value: "SSN" },
          { label: "CREDIT_CARD", value: "CREDIT_CARD" },
          { label: "OTHER", value: "OTHER" },
        ]}
      />
      <Button
        type="link"
        icon={<CheckOutlined />}
        loading={loading}
        onClick={() => onPromote(rule.id, label)}
      >
        Promote
      </Button>
    </Space>
  );
};

export default PendingRulesPage;
