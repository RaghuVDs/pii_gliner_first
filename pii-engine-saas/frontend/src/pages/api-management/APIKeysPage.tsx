import React, { useEffect, useState, useCallback } from "react";
import {
  Card,
  Table,
  Button,
  Space,
  Typography,
  Tag,
  Modal,
  Form,
  Input,
  Checkbox,
  InputNumber,
  Popconfirm,
  message,
  Alert,
} from "antd";
import {
  PlusOutlined,
  CopyOutlined,
  StopOutlined,
  SyncOutlined,
  KeyOutlined,
} from "@ant-design/icons";
import { apiKeysApi, CreateKeyRequest } from "@/api/apiKeys";
import { APIKey } from "@/types/models";
import dayjs from "dayjs";

const { Title, Text, Paragraph } = Typography;

const AVAILABLE_SCOPES = [
  "detect",
  "redact",
  "detect:async",
  "pii-types:read",
  "pii-types:write",
  "rules:read",
  "rules:write",
  "models:read",
  "models:write",
  "users:read",
  "users:write",
  "audit:read",
];

const APIKeysPage: React.FC = () => {
  const [keys, setKeys] = useState<APIKey[]>([]);
  const [loading, setLoading] = useState(true);
  const [createModalOpen, setCreateModalOpen] = useState(false);
  const [secretKey, setSecretKey] = useState<string | null>(null);
  const [form] = Form.useForm();

  const fetchKeys = useCallback(async () => {
    setLoading(true);
    try {
      const data = await apiKeysApi.listKeys();
      setKeys(data);
    } catch {
      message.error("Failed to load API keys");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchKeys();
  }, [fetchKeys]);

  const handleCreate = async (values: CreateKeyRequest) => {
    try {
      const result = await apiKeysApi.createKey(values);
      setSecretKey(result.secret_key);
      setCreateModalOpen(false);
      form.resetFields();
      message.success("API key created");
      fetchKeys();
    } catch {
      message.error("Failed to create API key");
    }
  };

  const handleRevoke = async (keyId: string) => {
    try {
      await apiKeysApi.revokeKey(keyId);
      message.success("API key revoked");
      fetchKeys();
    } catch {
      message.error("Failed to revoke API key");
    }
  };

  const handleRotate = async (keyId: string) => {
    try {
      const result = await apiKeysApi.rotateKey(keyId);
      setSecretKey(result.secret_key);
      message.success("API key rotated");
      fetchKeys();
    } catch {
      message.error("Failed to rotate API key");
    }
  };

  const columns = [
    {
      title: "Name",
      dataIndex: "name",
      key: "name",
      render: (name: string) => (
        <Space>
          <KeyOutlined />
          <Text strong>{name}</Text>
        </Space>
      ),
    },
    {
      title: "Key Prefix",
      dataIndex: "key_prefix",
      key: "key_prefix",
      render: (prefix: string) => <Text code>{prefix}...</Text>,
    },
    {
      title: "Scopes",
      dataIndex: "scopes",
      key: "scopes",
      render: (scopes: string[]) => (
        <Space size={4} wrap>
          {scopes.map((s) => (
            <Tag key={s} size="small" >{s}</Tag>
          ))}
        </Space>
      ),
    },
    {
      title: "Rate Limit",
      dataIndex: "rate_limit",
      key: "rate_limit",
      render: (limit: number) => `${limit}/min`,
    },
    {
      title: "Last Used",
      dataIndex: "last_used_at",
      key: "last_used_at",
      render: (date: string | null) =>
        date ? dayjs(date).format("MMM DD, HH:mm") : "Never",
    },
    {
      title: "Status",
      dataIndex: "is_active",
      key: "is_active",
      render: (active: boolean, record: APIKey) => {
        if (!active) return <Tag color="red">Revoked</Tag>;
        if (record.expires_at && dayjs(record.expires_at).isBefore(dayjs())) {
          return <Tag color="orange">Expired</Tag>;
        }
        return <Tag color="green">Active</Tag>;
      },
    },
    {
      title: "Actions",
      key: "actions",
      render: (_: unknown, record: APIKey) => (
        <Space size="small">
          <Button
            type="text"
            icon={<SyncOutlined />}
            size="small"
            onClick={() => handleRotate(record.id)}
            disabled={!record.is_active}
          >
            Rotate
          </Button>
          <Popconfirm
            title="Revoke this API key?"
            onConfirm={() => handleRevoke(record.id)}
          >
            <Button
              type="text"
              icon={<StopOutlined />}
              size="small"
              danger
              disabled={!record.is_active}
            >
              Revoke
            </Button>
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
          API Keys
        </Title>
        <Button
          type="primary"
          icon={<PlusOutlined />}
          onClick={() => setCreateModalOpen(true)}
        >
          Create Key
        </Button>
      </div>

      <Card>
        <Table
          columns={columns}
          dataSource={keys}
          rowKey="id"
          loading={loading}
          size="small"
          pagination={{ pageSize: 10 }}
        />
      </Card>

      {/* Create Key Modal */}
      <Modal
        title="Create API Key"
        open={createModalOpen}
        onCancel={() => setCreateModalOpen(false)}
        onOk={() => form.submit()}
        width={500}
      >
        <Form form={form} layout="vertical" onFinish={handleCreate}>
          <Form.Item
            name="name"
            label="Key Name"
            rules={[{ required: true, message: "Required" }]}
          >
            <Input placeholder="e.g. Production API Key" />
          </Form.Item>
          <Form.Item
            name="scopes"
            label="Scopes"
            rules={[{ required: true, message: "Select at least one scope" }]}
          >
            <Checkbox.Group
              options={AVAILABLE_SCOPES.map((s) => ({
                label: s,
                value: s,
              }))}
            />
          </Form.Item>
          <Form.Item
            name="rate_limit"
            label="Rate Limit (per minute)"
            initialValue={60}
          >
            <InputNumber min={1} max={10000} />
          </Form.Item>
          <Form.Item
            name="expires_in_days"
            label="Expires In (days)"
            extra="Leave empty for no expiration"
          >
            <InputNumber min={1} max={365} />
          </Form.Item>
        </Form>
      </Modal>

      {/* Secret Key Display Modal */}
      <Modal
        title="API Key Created"
        open={!!secretKey}
        onCancel={() => setSecretKey(null)}
        onOk={() => setSecretKey(null)}
        footer={[
          <Button key="close" onClick={() => setSecretKey(null)}>
            I&apos;ve saved it
          </Button>,
        ]}
      >
        <Alert
          type="warning"
          message="Save your API key now. You won't be able to see it again!"
          style={{ marginBottom: 16 }}
        />
        <div
          style={{
            background: "#f5f5f5",
            padding: 16,
            borderRadius: 8,
            display: "flex",
            alignItems: "center",
            gap: 8,
          }}
        >
          <Paragraph
            copyable
            style={{ margin: 0, flex: 1, fontFamily: "monospace" }}
          >
            {secretKey}
          </Paragraph>
          <Button
            icon={<CopyOutlined />}
            onClick={() => {
              if (secretKey) {
                navigator.clipboard.writeText(secretKey);
                message.success("Copied to clipboard");
              }
            }}
          />
        </div>
      </Modal>
    </div>
  );
};

export default APIKeysPage;
