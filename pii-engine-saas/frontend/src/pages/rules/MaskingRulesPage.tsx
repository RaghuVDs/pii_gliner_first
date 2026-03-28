import React, { useEffect, useState } from "react";
import {
  Card,
  Table,
  Select,
  Typography,
  Tag,
  Space,
  message,
  Spin,
} from "antd";
import { rulesApi } from "@/api/rules";
import { MaskingRule, MaskingStrategy } from "@/types/models";
import { MaskingStrategyType } from "@/types/enums";

const { Title, Text } = Typography;

const STRATEGY_COLORS: Record<MaskingStrategyType, string> = {
  [MaskingStrategyType.REDACT]: "red",
  [MaskingStrategyType.MASK]: "orange",
  [MaskingStrategyType.HASH]: "blue",
  [MaskingStrategyType.ENCRYPT]: "purple",
  [MaskingStrategyType.TOKENIZE]: "cyan",
  [MaskingStrategyType.GENERALIZE]: "green",
  [MaskingStrategyType.PLACEHOLDER]: "default",
};

const STRATEGY_PREVIEWS: Record<MaskingStrategyType, string> = {
  [MaskingStrategyType.REDACT]: "[REDACTED]",
  [MaskingStrategyType.MASK]: "J*** D**",
  [MaskingStrategyType.HASH]: "a1b2c3d4....",
  [MaskingStrategyType.ENCRYPT]: "enc:xK9mQ2...",
  [MaskingStrategyType.TOKENIZE]: "tok_12345",
  [MaskingStrategyType.GENERALIZE]: "<PERSON>",
  [MaskingStrategyType.PLACEHOLDER]: "***",
};

const MaskingRulesPage: React.FC = () => {
  const [rules, setRules] = useState<MaskingRule[]>([]);
  const [strategies, setStrategies] = useState<MaskingStrategy[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      try {
        const [rulesData, strategiesData] = await Promise.all([
          rulesApi.listMaskingRules(),
          rulesApi.listMaskingStrategies(),
        ]);
        setRules(rulesData);
        setStrategies(strategiesData);
      } catch {
        message.error("Failed to load masking configuration");
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  const handleStrategyChange = async (
    piiTypeName: string,
    strategyId: string
  ) => {
    try {
      await rulesApi.updateMaskingRule(piiTypeName, {
        strategy_id: strategyId,
      });
      message.success(`Masking strategy updated for ${piiTypeName}`);
      // Refresh
      const updated = await rulesApi.listMaskingRules();
      setRules(updated);
    } catch {
      message.error("Failed to update masking strategy");
    }
  };

  const columns = [
    {
      title: "PII Type",
      dataIndex: "pii_type_name",
      key: "pii_type_name",
      render: (name: string) => <Tag color="blue">{name}</Tag>,
    },
    {
      title: "Current Strategy",
      dataIndex: "strategy_name",
      key: "strategy_name",
      render: (name: string) => (
        <Tag color={name ? "blue" : "default"}>
          {name ? name.toUpperCase().replace(/_/g, " ") : "DEFAULT"}
        </Tag>
      ),
    },
    {
      title: "Change Strategy",
      key: "change",
      width: 240,
      render: (_: unknown, record: MaskingRule) => (
        <Select
          style={{ width: 200 }}
          value={record.strategy_id}
          onChange={(value) =>
            handleStrategyChange(record.pii_type_name, value)
          }
          options={strategies.map((s) => ({
            label: (
              <Space>
                <Tag
                  color={STRATEGY_COLORS[s.strategy_type] || "default"}
                >
                  {s.strategy_type}
                </Tag>
                {s.name}
              </Space>
            ),
            value: s.id,
          }))}
        />
      ),
    },
    {
      title: "Preview",
      key: "preview",
      render: (_: unknown, record: MaskingRule) => (
        <Text code>
          {STRATEGY_PREVIEWS[record.strategy_type] || "***"}
        </Text>
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
        Masking Rules
      </Title>

      <Card
        style={{ marginBottom: 16 }}
        title="Masking Strategies Reference"
        size="small"
      >
        <Space wrap>
          {strategies.map((s) => (
            <Tag
              key={s.id}
              color={STRATEGY_COLORS[s.strategy_type] || "default"}
            >
              {s.strategy_type}: {s.description}
            </Tag>
          ))}
        </Space>
      </Card>

      <Card>
        <Table
          columns={columns}
          dataSource={rules}
          rowKey="id"
          size="middle"
          pagination={false}
        />
      </Card>
    </div>
  );
};

export default MaskingRulesPage;
