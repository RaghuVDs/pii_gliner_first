import React, { useState } from "react";
import {
  Card,
  Form,
  Input,
  Button,
  Typography,
  Space,
  Divider,
  Avatar,
  Tag,
  message,
} from "antd";
import { UserOutlined, SaveOutlined } from "@ant-design/icons";
import { useAuthStore } from "@/store/authStore";
import { usersApi, UpdateProfileRequest } from "@/api/users";
import { UserRole } from "@/types/enums";

const { Title, Text } = Typography;

const ROLE_COLORS: Record<UserRole, string> = {
  [UserRole.SUPER_ADMIN]: "red",
  [UserRole.TENANT_ADMIN]: "orange",
  [UserRole.ANALYST]: "blue",
  [UserRole.API_USER]: "cyan",
  [UserRole.VIEWER]: "default",
};

const ProfilePage: React.FC = () => {
  const user = useAuthStore((state) => state.user);
  const setUser = useAuthStore((state) => state.setUser);
  const [loading, setLoading] = useState(false);
  const [passwordLoading, setPasswordLoading] = useState(false);
  const [profileForm] = Form.useForm();
  const [passwordForm] = Form.useForm();

  const handleProfileUpdate = async (values: { full_name: string }) => {
    setLoading(true);
    try {
      const updated = await usersApi.updateProfile({
        full_name: values.full_name,
      });
      setUser(updated);
      message.success("Profile updated");
    } catch {
      message.error("Failed to update profile");
    } finally {
      setLoading(false);
    }
  };

  const handlePasswordChange = async (values: {
    current_password: string;
    new_password: string;
  }) => {
    setPasswordLoading(true);
    try {
      const data: UpdateProfileRequest = {
        current_password: values.current_password,
        new_password: values.new_password,
      };
      await usersApi.updateProfile(data);
      message.success("Password changed successfully");
      passwordForm.resetFields();
    } catch {
      message.error("Failed to change password");
    } finally {
      setPasswordLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: 700 }}>
      <Title level={4} style={{ marginBottom: 24 }}>
        Profile
      </Title>

      {/* User Info Card */}
      <Card style={{ marginBottom: 24 }}>
        <Space size="large" align="start">
          <Avatar
            size={72}
            icon={<UserOutlined />}
            style={{ backgroundColor: "#1677ff" }}
          />
          <div>
            <Title level={5} style={{ margin: 0 }}>
              {user?.full_name}
            </Title>
            <Text type="secondary">{user?.email}</Text>
            <br />
            <Tag
              color={ROLE_COLORS[user?.role || UserRole.VIEWER]}
              style={{ marginTop: 8 }}
            >
              {(user?.role || "viewer").replace("_", " ").toUpperCase()}
            </Tag>
          </div>
        </Space>
      </Card>

      {/* Profile Form */}
      <Card title="Edit Profile" style={{ marginBottom: 24 }}>
        <Form
          form={profileForm}
          layout="vertical"
          onFinish={handleProfileUpdate}
          initialValues={{
            full_name: user?.full_name || "",
            email: user?.email || "",
          }}
        >
          <Form.Item name="email" label="Email">
            <Input disabled />
          </Form.Item>
          <Form.Item
            name="full_name"
            label="Full Name"
            rules={[{ required: true, message: "Required" }]}
          >
            <Input />
          </Form.Item>
          <Form.Item>
            <Button
              type="primary"
              htmlType="submit"
              icon={<SaveOutlined />}
              loading={loading}
            >
              Save Changes
            </Button>
          </Form.Item>
        </Form>
      </Card>

      {/* Change Password */}
      <Card title="Change Password">
        <Form
          form={passwordForm}
          layout="vertical"
          onFinish={handlePasswordChange}
        >
          <Form.Item
            name="current_password"
            label="Current Password"
            rules={[{ required: true, message: "Required" }]}
          >
            <Input.Password />
          </Form.Item>
          <Form.Item
            name="new_password"
            label="New Password"
            rules={[
              { required: true, message: "Required" },
              { min: 8, message: "Min 8 characters" },
            ]}
          >
            <Input.Password />
          </Form.Item>
          <Form.Item
            name="confirm_password"
            label="Confirm New Password"
            dependencies={["new_password"]}
            rules={[
              { required: true, message: "Required" },
              ({ getFieldValue }) => ({
                validator(_, value) {
                  if (!value || getFieldValue("new_password") === value) {
                    return Promise.resolve();
                  }
                  return Promise.reject(
                    new Error("Passwords do not match")
                  );
                },
              }),
            ]}
          >
            <Input.Password />
          </Form.Item>

          <Divider />

          <Form.Item>
            <Button
              type="primary"
              htmlType="submit"
              loading={passwordLoading}
            >
              Change Password
            </Button>
          </Form.Item>
        </Form>
      </Card>
    </div>
  );
};

export default ProfilePage;
