import React, { useState } from "react";
import { Link } from "react-router-dom";
import { Form, Input, Button, Card, Typography, Space, Divider } from "antd";
import {
  MailOutlined,
  LockOutlined,
  UserOutlined,
  BankOutlined,
  ScanOutlined,
} from "@ant-design/icons";
import { useAuth } from "@/hooks/useAuth";

const { Title, Text } = Typography;

interface RegisterFormValues {
  email: string;
  password: string;
  confirmPassword: string;
  fullName: string;
  orgName: string;
}

const RegisterPage: React.FC = () => {
  const [loading, setLoading] = useState(false);
  const { register } = useAuth();
  const [form] = Form.useForm<RegisterFormValues>();

  const handleSubmit = async (values: RegisterFormValues) => {
    setLoading(true);
    try {
      await register(
        values.email,
        values.password,
        values.fullName,
        values.orgName
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      style={{
        minHeight: "100vh",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
        padding: 24,
      }}
    >
      <Card
        style={{
          width: 480,
          boxShadow: "0 8px 24px rgba(0, 0, 0, 0.15)",
          borderRadius: 12,
        }}
      >
        <Space
          direction="vertical"
          size="large"
          style={{ width: "100%", textAlign: "center" }}
        >
          <div>
            <ScanOutlined
              style={{ fontSize: 48, color: "#1677ff", marginBottom: 8 }}
            />
            <Title level={3} style={{ margin: 0 }}>
              Create Account
            </Title>
            <Text type="secondary">
              Get started with PII Engine
            </Text>
          </div>

          <Form
            form={form}
            layout="vertical"
            onFinish={handleSubmit}
            autoComplete="off"
            size="large"
          >
            <Form.Item
              name="fullName"
              rules={[
                { required: true, message: "Please enter your full name" },
              ]}
            >
              <Input prefix={<UserOutlined />} placeholder="Full name" />
            </Form.Item>

            <Form.Item
              name="orgName"
              rules={[
                {
                  required: true,
                  message: "Please enter your organization name",
                },
              ]}
            >
              <Input
                prefix={<BankOutlined />}
                placeholder="Organization name"
              />
            </Form.Item>

            <Form.Item
              name="email"
              rules={[
                { required: true, message: "Please enter your email" },
                { type: "email", message: "Please enter a valid email" },
              ]}
            >
              <Input
                prefix={<MailOutlined />}
                placeholder="Email address"
              />
            </Form.Item>

            <Form.Item
              name="password"
              rules={[
                { required: true, message: "Please enter a password" },
                {
                  min: 8,
                  message: "Password must be at least 8 characters",
                },
              ]}
            >
              <Input.Password
                prefix={<LockOutlined />}
                placeholder="Password"
              />
            </Form.Item>

            <Form.Item
              name="confirmPassword"
              dependencies={["password"]}
              rules={[
                { required: true, message: "Please confirm your password" },
                ({ getFieldValue }) => ({
                  validator(_, value) {
                    if (!value || getFieldValue("password") === value) {
                      return Promise.resolve();
                    }
                    return Promise.reject(
                      new Error("Passwords do not match")
                    );
                  },
                }),
              ]}
            >
              <Input.Password
                prefix={<LockOutlined />}
                placeholder="Confirm password"
              />
            </Form.Item>

            <Form.Item>
              <Button
                type="primary"
                htmlType="submit"
                loading={loading}
                block
              >
                Create Account
              </Button>
            </Form.Item>
          </Form>

          <Divider plain>
            <Text type="secondary" style={{ fontSize: 12 }}>
              or
            </Text>
          </Divider>

          <Text>
            Already have an account?{" "}
            <Link to="/login">
              <Text strong style={{ color: "#1677ff" }}>
                Sign in
              </Text>
            </Link>
          </Text>
        </Space>
      </Card>
    </div>
  );
};

export default RegisterPage;
