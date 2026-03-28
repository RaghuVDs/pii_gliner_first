import React, { useState } from "react";
import { Link } from "react-router-dom";
import { Form, Input, Button, Card, Typography, Space, Divider } from "antd";
import { MailOutlined, LockOutlined, ScanOutlined } from "@ant-design/icons";
import { useAuth } from "@/hooks/useAuth";

const { Title, Text } = Typography;

interface LoginFormValues {
  email: string;
  password: string;
}

const LoginPage: React.FC = () => {
  const [loading, setLoading] = useState(false);
  const { login } = useAuth();
  const [form] = Form.useForm<LoginFormValues>();

  const handleSubmit = async (values: LoginFormValues) => {
    setLoading(true);
    try {
      await login(values.email, values.password);
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
          width: 420,
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
              PII Engine
            </Title>
            <Text type="secondary">
              Enterprise PII Detection Platform
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
                { required: true, message: "Please enter your password" },
              ]}
            >
              <Input.Password
                prefix={<LockOutlined />}
                placeholder="Password"
              />
            </Form.Item>

            <Form.Item>
              <Button
                type="primary"
                htmlType="submit"
                loading={loading}
                block
              >
                Sign In
              </Button>
            </Form.Item>
          </Form>

          <div style={{ textAlign: "right", marginTop: -16 }}>
            <Link to="/forgot-password">
              <Text type="secondary" style={{ fontSize: 13 }}>
                Forgot password?
              </Text>
            </Link>
          </div>

          <Divider plain>
            <Text type="secondary" style={{ fontSize: 12 }}>
              or
            </Text>
          </Divider>

          <Text>
            Don&apos;t have an account?{" "}
            <Link to="/register">
              <Text strong style={{ color: "#1677ff" }}>
                Sign up
              </Text>
            </Link>
          </Text>
        </Space>
      </Card>
    </div>
  );
};

export default LoginPage;
