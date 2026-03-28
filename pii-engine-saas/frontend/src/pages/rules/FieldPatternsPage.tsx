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
} from "antd";
import {
  PlusOutlined,
  EditOutlined,
  DeleteOutlined,
  LockOutlined,
} from "@ant-design/icons";
import { rulesApi } from "@/api/rules";
import { FieldPattern } from "@/types/models";

const { Title } = Typography;

const FieldPatternsPage: React.FC = () => {
  const [patterns, setPatterns] = useState<FieldPattern[]>([]);
  const [loading, setLoading] = useState(true);
  const [filterType, setFilterType] = useState<string | undefined>();
  const [modalOpen, setModalOpen] = useState(false);
  const [editingPattern, setEditingPattern] = useState<FieldPattern | null>(
    null
  );
  const [form] = Form.useForm();

  const fetchPatterns = useCallback(async () => {
    setLoading(true);
    try {
      const data = await rulesApi.listFieldPatterns(filterType);
      setPatterns(data);
    } catch {
      message.error("Failed to load field patterns");
    } finally {
      setLoading(false);
    }
  }, [filterType]);

  useEffect(() => {
    fetchPatterns();
  }, [fetchPatterns]);

  const handleSave = async (values: Partial<FieldPattern>) => {
    try {
      if (editingPattern) {
        await rulesApi.updateFieldPattern(editingPattern.id, values);
        message.success("Field pattern updated");
      } else {
        await rulesApi.createFieldPattern(values);
        message.success("Field pattern created");
      }
      setModalOpen(false);
      setEditingPattern(null);
      form.resetFields();
      fetchPatterns();
    } catch {
      message.error("Failed to save field pattern");
    }
  };

  const handleDelete = async (id: string) => {
    try {
      await rulesApi.deleteFieldPattern(id);
      message.success("Field pattern deleted");
      fetchPatterns();
    } catch {
      message.error("Failed to delete field pattern");
    }
  };

  const openEdit = (pattern: FieldPattern) => {
    setEditingPattern(pattern);
    form.setFieldsValue(pattern);
    setModalOpen(true);
  };

  const piiTypeNames = [...new Set(patterns.map((p) => p.pii_type_name))];

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
        <code style={{ fontSize: 12 }}>{pattern}</code>
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
      render: (name: string) => <Tag color="cyan">{name}</Tag>,
    },
    {
      title: "Active",
      dataIndex: "is_active",
      key: "is_active",
      width: 80,
      render: (active: boolean, record: FieldPattern) => (
        <Switch
          size="small"
          checked={active}
          disabled={record.is_system}
          onChange={async (checked) => {
            await rulesApi.updateFieldPattern(record.id, {
              is_active: checked,
            });
            fetchPatterns();
          }}
        />
      ),
    },
    {
      title: "Actions",
      key: "actions",
      width: 100,
      render: (_: unknown, record: FieldPattern) =>
        record.is_system ? null : (
          <Space size="small">
            <Button
              type="text"
              icon={<EditOutlined />}
              size="small"
              onClick={() => openEdit(record)}
            />
            <Popconfirm
              title="Delete this field pattern?"
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
          Field Patterns
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
              setEditingPattern(null);
              form.resetFields();
              setModalOpen(true);
            }}
          >
            Add Pattern
          </Button>
        </Space>
      </div>

      <Card>
        <Table
          columns={columns}
          dataSource={patterns}
          rowKey="id"
          loading={loading}
          size="small"
          pagination={{ pageSize: 20 }}
        />
      </Card>

      <Modal
        title={
          editingPattern ? "Edit Field Pattern" : "Add Field Pattern"
        }
        open={modalOpen}
        onCancel={() => {
          setModalOpen(false);
          setEditingPattern(null);
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
            label="Field Pattern"
            rules={[{ required: true, message: "Required" }]}
            extra="Pattern to match field names (e.g., *email*, user_mail)"
          >
            <Input
              placeholder="e.g. *email*, *mail_address*"
              style={{ fontFamily: "monospace" }}
            />
          </Form.Item>
          <Form.Item name="description" label="Description">
            <Input placeholder="Brief description" />
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

export default FieldPatternsPage;
