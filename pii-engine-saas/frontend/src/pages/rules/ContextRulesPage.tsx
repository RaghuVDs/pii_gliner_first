import React, { useEffect, useState, useCallback } from "react";
import {
  Card,
  Table,
  Button,
  Space,
  Typography,
  Modal,
  Form,
  Input,
  Select,
  Tag,
  Switch,
  Popconfirm,
  InputNumber,
  message,
} from "antd";
import {
  PlusOutlined,
  EditOutlined,
  DeleteOutlined,
  LockOutlined,
} from "@ant-design/icons";
import { rulesApi } from "@/api/rules";
import { ContextRule } from "@/types/models";

const { Title } = Typography;

const ContextRulesPage: React.FC = () => {
  const [rules, setRules] = useState<ContextRule[]>([]);
  const [loading, setLoading] = useState(true);
  const [filterType, setFilterType] = useState<string | undefined>();
  const [modalOpen, setModalOpen] = useState(false);
  const [editingRule, setEditingRule] = useState<ContextRule | null>(null);
  const [form] = Form.useForm();

  const fetchRules = useCallback(async () => {
    setLoading(true);
    try {
      const data = await rulesApi.listContextRules(filterType);
      setRules(data);
    } catch {
      message.error("Failed to load context rules");
    } finally {
      setLoading(false);
    }
  }, [filterType]);

  useEffect(() => {
    fetchRules();
  }, [fetchRules]);

  const handleSave = async (values: Record<string, unknown>) => {
    const payload = {
      ...values,
      positive_contexts:
        typeof values.positive_contexts === "string"
          ? (values.positive_contexts as string).split(",").map((s: string) => s.trim()).filter(Boolean)
          : values.positive_contexts,
      negative_contexts:
        typeof values.negative_contexts === "string"
          ? (values.negative_contexts as string).split(",").map((s: string) => s.trim()).filter(Boolean)
          : values.negative_contexts,
    };

    try {
      if (editingRule) {
        await rulesApi.updateContextRule(editingRule.id, payload);
        message.success("Context rule updated");
      } else {
        await rulesApi.createContextRule(payload);
        message.success("Context rule created");
      }
      setModalOpen(false);
      setEditingRule(null);
      form.resetFields();
      fetchRules();
    } catch {
      message.error("Failed to save context rule");
    }
  };

  const handleDelete = async (id: string) => {
    try {
      await rulesApi.deleteContextRule(id);
      message.success("Context rule deleted");
      fetchRules();
    } catch {
      message.error("Failed to delete context rule");
    }
  };

  const openEdit = (rule: ContextRule) => {
    setEditingRule(rule);
    form.setFieldsValue({
      ...rule,
      positive_contexts: rule.positive_contexts.join(", "),
      negative_contexts: rule.negative_contexts.join(", "),
    });
    setModalOpen(true);
  };

  const piiTypeNames = [...new Set(rules.map((r) => r.pii_type_name))];

  const columns = [
    {
      title: "",
      dataIndex: "is_system",
      key: "is_system",
      width: 30,
      render: (isSystem: boolean) =>
        isSystem ? <LockOutlined style={{ color: "#999" }} /> : null,
    },
    {
      title: "PII Type",
      dataIndex: "pii_type_name",
      key: "pii_type_name",
      render: (name: string) => <Tag color="orange">{name}</Tag>,
    },
    {
      title: "Keyword Pattern",
      dataIndex: "keyword_pattern",
      key: "keyword_pattern",
      render: (pattern: string) => <Tag color="blue">{pattern}</Tag>,
    },
    {
      title: "Type",
      dataIndex: "is_negative",
      key: "is_negative",
      render: (isNeg: boolean) => (
        <Tag color={isNeg ? "red" : "green"}>
          {isNeg ? "Negative" : "Positive"}
        </Tag>
      ),
    },
    {
      title: "Boost",
      dataIndex: "boost_score",
      key: "boost_score",
      width: 80,
      render: (score: number) => (
        <Tag color="green">+{score}</Tag>
      ),
    },
    {
      title: "Penalty",
      dataIndex: "penalty_score",
      key: "penalty_score",
      width: 80,
      render: (score: number) => (
        <Tag color="red">-{score}</Tag>
      ),
    },
    {
      title: "Active",
      dataIndex: "is_active",
      key: "is_active",
      width: 70,
      render: (active: boolean, record: ContextRule) => (
        <Switch
          size="small"
          checked={active}
          disabled={record.is_system}
          onChange={async (checked) => {
            await rulesApi.updateContextRule(record.id, {
              is_active: checked,
            });
            fetchRules();
          }}
        />
      ),
    },
    {
      title: "Actions",
      key: "actions",
      width: 100,
      render: (_: unknown, record: ContextRule) =>
        record.is_system ? null : (
          <Space size="small">
            <Button
              type="text"
              icon={<EditOutlined />}
              size="small"
              onClick={() => openEdit(record)}
            />
            <Popconfirm
              title="Delete this rule?"
              onConfirm={() => handleDelete(record.id)}
            >
              <Button
                type="text"
                icon={<DeleteOutlined />}
                size="small"
                danger
              />
            </Popconfirm>
          </Space>
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
          Context Rules
        </Title>
        <Space>
          <Select
            placeholder="Filter by PII type"
            style={{ width: 200 }}
            value={filterType}
            onChange={setFilterType}
            allowClear
            options={piiTypeNames.map((n) => ({ label: n, value: n }))}
          />
          <Button
            type="primary"
            icon={<PlusOutlined />}
            onClick={() => {
              setEditingRule(null);
              form.resetFields();
              setModalOpen(true);
            }}
          >
            Add Rule
          </Button>
        </Space>
      </div>

      <Card>
        <Table
          columns={columns}
          dataSource={rules}
          rowKey="id"
          loading={loading}
          size="small"
          pagination={{ pageSize: 20 }}
          scroll={{ x: 900 }}
        />
      </Card>

      <Modal
        title={editingRule ? "Edit Context Rule" : "Add Context Rule"}
        open={modalOpen}
        onCancel={() => {
          setModalOpen(false);
          setEditingRule(null);
        }}
        onOk={() => form.submit()}
        width={600}
      >
        <Form form={form} layout="vertical" onFinish={handleSave}>
          <Form.Item
            name="pii_type_name"
            label="PII Type"
            rules={[{ required: true, message: "Required" }]}
          >
            <Input placeholder="e.g. PERSON" />
          </Form.Item>
          <Form.Item
            name="positive_contexts"
            label="Positive Contexts"
            extra="Comma separated words that boost confidence"
          >
            <Input placeholder="e.g. name, patient, author, employee" />
          </Form.Item>
          <Form.Item
            name="negative_contexts"
            label="Negative Contexts"
            extra="Comma separated words that reduce confidence"
          >
            <Input placeholder="e.g. company, product, city" />
          </Form.Item>
          <Form.Item
            name="boost_score"
            label="Boost Score"
            initialValue={0.1}
          >
            <InputNumber min={0} max={1} step={0.05} />
          </Form.Item>
          <Form.Item
            name="penalty_score"
            label="Penalty Score"
            initialValue={0.1}
          >
            <InputNumber min={0} max={1} step={0.05} />
          </Form.Item>
          <Form.Item
            name="is_active"
            label="Active"
            valuePropName="checked"
            initialValue={true}
          >
            <Switch />
          </Form.Item>
        </Form>
      </Modal>
    </div>
  );
};

export default ContextRulesPage;
