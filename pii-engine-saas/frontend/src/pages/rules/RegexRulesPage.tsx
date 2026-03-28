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
  message,
  Row,
  Col,
  Divider,
} from "antd";
import {
  PlusOutlined,
  EditOutlined,
  DeleteOutlined,
  LockOutlined,
  PlayCircleOutlined,
} from "@ant-design/icons";
import { rulesApi } from "@/api/rules";
import { RegexRule } from "@/types/models";

const { Title, Text } = Typography;
const { TextArea } = Input;

const RegexRulesPage: React.FC = () => {
  const [rules, setRules] = useState<RegexRule[]>([]);
  const [loading, setLoading] = useState(true);
  const [filterType, setFilterType] = useState<string | undefined>();
  const [modalOpen, setModalOpen] = useState(false);
  const [editingRule, setEditingRule] = useState<RegexRule | null>(null);
  const [form] = Form.useForm();

  // Test panel state
  const [testPattern, setTestPattern] = useState("");
  const [testText, setTestText] = useState("");
  const [testMatches, setTestMatches] = useState<
    { text: string; start: number; end: number }[]
  >([]);
  const [testing, setTesting] = useState(false);

  const fetchRules = useCallback(async () => {
    setLoading(true);
    try {
      const data = await rulesApi.listRegexRules(filterType);
      setRules(data);
    } catch {
      message.error("Failed to load regex rules");
    } finally {
      setLoading(false);
    }
  }, [filterType]);

  useEffect(() => {
    fetchRules();
  }, [fetchRules]);

  const handleSave = async (values: Partial<RegexRule>) => {
    try {
      if (editingRule) {
        await rulesApi.updateRegexRule(editingRule.id, values);
        message.success("Rule updated");
      } else {
        await rulesApi.createRegexRule(values);
        message.success("Rule created");
      }
      setModalOpen(false);
      setEditingRule(null);
      form.resetFields();
      fetchRules();
    } catch {
      message.error("Failed to save rule");
    }
  };

  const handleDelete = async (id: string) => {
    try {
      await rulesApi.deleteRegexRule(id);
      message.success("Rule deleted");
      fetchRules();
    } catch {
      message.error("Failed to delete rule");
    }
  };

  const handleTest = async () => {
    if (!testPattern || !testText) {
      message.warning("Please enter both a pattern and test text");
      return;
    }
    setTesting(true);
    try {
      const result = await rulesApi.testRegexPattern(testPattern, testText);
      setTestMatches(result.matches);
      message.success(`Found ${result.matches.length} matches`);
    } catch {
      message.error("Pattern test failed");
    } finally {
      setTesting(false);
    }
  };

  const openEdit = (rule: RegexRule) => {
    setEditingRule(rule);
    form.setFieldsValue(rule);
    setModalOpen(true);
  };

  const openCreate = () => {
    setEditingRule(null);
    form.resetFields();
    setModalOpen(true);
  };

  // Group by PII type
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
      title: "Pattern",
      dataIndex: "pattern",
      key: "pattern",
      render: (pattern: string) => (
        <Text code style={{ fontSize: 12, wordBreak: "break-all" as const }}>
          {pattern}
        </Text>
      ),
    },
    {
      title: "Description",
      dataIndex: "description",
      key: "description",
      ellipsis: true,
    },
    {
      title: "PII Type",
      dataIndex: "pii_type_name",
      key: "pii_type_name",
      render: (name: string) => <Tag color="blue">{name}</Tag>,
    },
    {
      title: "Priority",
      dataIndex: "priority",
      key: "priority",
      width: 80,
    },
    {
      title: "Active",
      dataIndex: "is_active",
      key: "is_active",
      width: 80,
      render: (active: boolean, record: RegexRule) => (
        <Switch
          size="small"
          checked={active}
          disabled={record.is_system}
          onChange={async (checked) => {
            await rulesApi.updateRegexRule(record.id, { is_active: checked });
            fetchRules();
          }}
        />
      ),
    },
    {
      title: "Actions",
      key: "actions",
      width: 100,
      render: (_: unknown, record: RegexRule) =>
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
          Regex Rules
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
          <Button type="primary" icon={<PlusOutlined />} onClick={openCreate}>
            Add Rule
          </Button>
        </Space>
      </div>

      <Row gutter={16}>
        <Col xs={24} lg={16}>
          <Card>
            <Table
              columns={columns}
              dataSource={rules}
              rowKey="id"
              loading={loading}
              size="small"
              pagination={{ pageSize: 20 }}
            />
          </Card>
        </Col>

        {/* Test Panel */}
        <Col xs={24} lg={8}>
          <Card title="Test Regex Pattern">
            <Space direction="vertical" style={{ width: "100%" }} size="middle">
              <Input
                placeholder="Regex pattern"
                value={testPattern}
                onChange={(e) => setTestPattern(e.target.value)}
                addonBefore="/"
                addonAfter="/"
              />
              <TextArea
                rows={4}
                placeholder="Test text..."
                value={testText}
                onChange={(e) => setTestText(e.target.value)}
              />
              <Button
                type="primary"
                icon={<PlayCircleOutlined />}
                onClick={handleTest}
                loading={testing}
                block
              >
                Test Pattern
              </Button>

              {testMatches.length > 0 && (
                <>
                  <Divider />
                  <Text strong>
                    {testMatches.length} match(es) found:
                  </Text>
                  {testMatches.map((m, i) => (
                    <Tag key={i} color="green">
                      &quot;{m.text}&quot; [{m.start}:{m.end}]
                    </Tag>
                  ))}
                </>
              )}
            </Space>
          </Card>
        </Col>
      </Row>

      {/* Add/Edit Modal */}
      <Modal
        title={editingRule ? "Edit Regex Rule" : "Add Regex Rule"}
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
            <Input placeholder="e.g. EMAIL" />
          </Form.Item>
          <Form.Item
            name="pattern"
            label="Regex Pattern"
            rules={[{ required: true, message: "Required" }]}
          >
            <Input.TextArea
              rows={3}
              placeholder="Enter regex pattern"
              style={{ fontFamily: "monospace" }}
            />
          </Form.Item>
          <Form.Item name="description" label="Description">
            <Input placeholder="Brief description of what this pattern matches" />
          </Form.Item>
          <Form.Item name="priority" label="Priority" initialValue={0}>
            <Input type="number" />
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

export default RegexRulesPage;
