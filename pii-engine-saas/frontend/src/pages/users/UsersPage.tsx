import React, { useEffect, useState, useCallback } from "react";
import {
  Card,
  Table,
  Button,
  Space,
  Typography,
  Tag,
  Select,
  Modal,
  Form,
  Input,
  Popconfirm,
  Switch,
  message,
} from "antd";
import {
  PlusOutlined,
  EditOutlined,
  DeleteOutlined,
  UserOutlined,
} from "@ant-design/icons";
import { usersApi, CreateUserRequest, UpdateUserRequest } from "@/api/users";
import { User } from "@/types/models";
import { UserRole } from "@/types/enums";
import { usePagination } from "@/hooks/usePagination";
import dayjs from "dayjs";

const { Title, Text } = Typography;

const ROLE_COLORS: Record<UserRole, string> = {
  [UserRole.SUPER_ADMIN]: "red",
  [UserRole.TENANT_ADMIN]: "orange",
  [UserRole.ANALYST]: "blue",
  [UserRole.API_USER]: "cyan",
  [UserRole.VIEWER]: "default",
};

const UsersPage: React.FC = () => {
  const [users, setUsers] = useState<User[]>([]);
  const [loading, setLoading] = useState(true);
  const [filterRole, setFilterRole] = useState<UserRole | undefined>();
  const [modalOpen, setModalOpen] = useState(false);
  const [editingUser, setEditingUser] = useState<User | null>(null);
  const [form] = Form.useForm();
  const { page, pageSize, setTotal, paginationProps } = usePagination();

  const fetchUsers = useCallback(async () => {
    setLoading(true);
    try {
      const data = await usersApi.listUsers(page, pageSize, filterRole);
      setUsers(data.items);
      setTotal(data.total);
    } catch {
      message.error("Failed to load users");
    } finally {
      setLoading(false);
    }
  }, [page, pageSize, filterRole, setTotal]);

  useEffect(() => {
    fetchUsers();
  }, [fetchUsers]);

  const handleCreate = async (values: CreateUserRequest) => {
    try {
      await usersApi.createUser(values);
      message.success("User invited successfully");
      setModalOpen(false);
      form.resetFields();
      fetchUsers();
    } catch {
      message.error("Failed to create user");
    }
  };

  const handleUpdate = async (values: UpdateUserRequest) => {
    if (!editingUser) return;
    try {
      await usersApi.updateUser(editingUser.id, values);
      message.success("User updated");
      setModalOpen(false);
      setEditingUser(null);
      form.resetFields();
      fetchUsers();
    } catch {
      message.error("Failed to update user");
    }
  };

  const handleDelete = async (id: string) => {
    try {
      await usersApi.deleteUser(id);
      message.success("User deleted");
      fetchUsers();
    } catch {
      message.error("Failed to delete user");
    }
  };

  const openEdit = (user: User) => {
    setEditingUser(user);
    form.setFieldsValue({
      full_name: user.full_name,
      role: user.role,
      is_active: user.is_active,
    });
    setModalOpen(true);
  };

  const columns = [
    {
      title: "User",
      key: "user",
      render: (_: unknown, record: User) => (
        <Space>
          <UserOutlined />
          <div>
            <Text strong>{record.full_name}</Text>
            <br />
            <Text type="secondary" style={{ fontSize: 12 }}>
              {record.email}
            </Text>
          </div>
        </Space>
      ),
    },
    {
      title: "Role",
      dataIndex: "role",
      key: "role",
      render: (role: UserRole) => (
        <Tag color={ROLE_COLORS[role]}>{role.replace("_", " ").toUpperCase()}</Tag>
      ),
    },
    {
      title: "Status",
      dataIndex: "is_active",
      key: "is_active",
      render: (active: boolean) => (
        <Tag color={active ? "green" : "red"}>
          {active ? "Active" : "Inactive"}
        </Tag>
      ),
    },
    {
      title: "Last Login",
      dataIndex: "last_login",
      key: "last_login",
      render: (date: string | null) =>
        date ? dayjs(date).format("MMM DD, YYYY HH:mm") : "Never",
    },
    {
      title: "Created",
      dataIndex: "created_at",
      key: "created_at",
      render: (date: string) => dayjs(date).format("MMM DD, YYYY"),
    },
    {
      title: "Actions",
      key: "actions",
      render: (_: unknown, record: User) => (
        <Space size="small">
          <Button
            type="text"
            icon={<EditOutlined />}
            size="small"
            onClick={() => openEdit(record)}
          />
          <Popconfirm
            title="Delete this user?"
            description="This action cannot be undone."
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
          Users
        </Title>
        <Space>
          <Select
            placeholder="Filter by role"
            style={{ width: 160 }}
            value={filterRole}
            onChange={setFilterRole}
            allowClear
            options={Object.values(UserRole).map((r) => ({
              label: r.replace("_", " ").toUpperCase(),
              value: r,
            }))}
          />
          <Button
            type="primary"
            icon={<PlusOutlined />}
            onClick={() => {
              setEditingUser(null);
              form.resetFields();
              setModalOpen(true);
            }}
          >
            Invite User
          </Button>
        </Space>
      </div>

      <Card>
        <Table
          columns={columns}
          dataSource={users}
          rowKey="id"
          loading={loading}
          size="middle"
          pagination={paginationProps}
        />
      </Card>

      {/* Create/Edit Modal */}
      <Modal
        title={editingUser ? "Edit User" : "Invite User"}
        open={modalOpen}
        onCancel={() => {
          setModalOpen(false);
          setEditingUser(null);
        }}
        onOk={() => form.submit()}
        width={500}
      >
        <Form
          form={form}
          layout="vertical"
          onFinish={editingUser ? handleUpdate : handleCreate}
        >
          {!editingUser && (
            <>
              <Form.Item
                name="email"
                label="Email"
                rules={[
                  { required: true, message: "Required" },
                  { type: "email", message: "Valid email required" },
                ]}
              >
                <Input placeholder="user@company.com" />
              </Form.Item>
              <Form.Item
                name="password"
                label="Temporary Password"
                rules={[
                  { required: true, message: "Required" },
                  { min: 8, message: "Min 8 characters" },
                ]}
              >
                <Input.Password />
              </Form.Item>
            </>
          )}
          <Form.Item
            name="full_name"
            label="Full Name"
            rules={[{ required: true, message: "Required" }]}
          >
            <Input />
          </Form.Item>
          <Form.Item
            name="role"
            label="Role"
            rules={[{ required: true, message: "Required" }]}
          >
            <Select
              options={Object.values(UserRole).map((r) => ({
                label: r.replace("_", " ").toUpperCase(),
                value: r,
              }))}
            />
          </Form.Item>
          {editingUser && (
            <Form.Item
              name="is_active"
              label="Active"
              valuePropName="checked"
            >
              <Switch />
            </Form.Item>
          )}
        </Form>
      </Modal>
    </div>
  );
};

export default UsersPage;
