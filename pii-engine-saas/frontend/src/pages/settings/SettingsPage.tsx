import React from "react";
import { useParams, useNavigate } from "react-router-dom";
import {
  Card,
  Tabs,
  Form,
  Input,
  Switch,
  Button,
  Select,
  Typography,
  Space,
  Divider,
  Upload,
  message,
} from "antd";
import {
  UploadOutlined,
  DownloadOutlined,
} from "@ant-design/icons";
import { useTenantStore } from "@/store/tenantStore";

const { Title, Text } = Typography;

const SettingsPage: React.FC = () => {
  const { tab } = useParams<{ tab?: string }>();
  const navigate = useNavigate();
  const { tenant, updateTenant } = useTenantStore();
  const [generalForm] = Form.useForm();

  const activeTab = tab || "general";

  const handleTabChange = (key: string) => {
    navigate(`/settings/${key}`);
  };

  const handleGeneralSave = async (values: Record<string, unknown>) => {
    try {
      await updateTenant({
        name: values.name as string,
        settings: {
          ...tenant?.settings,
          ...values,
        },
      });
      message.success("Settings saved");
    } catch {
      message.error("Failed to save settings");
    }
  };

  const tabItems = [
    {
      key: "general",
      label: "General",
      children: (
        <Card>
          <Form
            form={generalForm}
            layout="vertical"
            onFinish={handleGeneralSave}
            initialValues={{
              name: tenant?.name || "",
              default_language: "en",
              auto_retrain: true,
              retrain_threshold: 100,
            }}
            style={{ maxWidth: 600 }}
          >
            <Form.Item
              name="name"
              label="Organization Name"
              rules={[{ required: true }]}
            >
              <Input />
            </Form.Item>

            <Form.Item
              name="default_language"
              label="Default Language"
            >
              <Select
                options={[
                  { label: "English", value: "en" },
                  { label: "Spanish", value: "es" },
                  { label: "French", value: "fr" },
                  { label: "German", value: "de" },
                ]}
              />
            </Form.Item>

            <Divider />

            <Title level={5}>Detection Settings</Title>

            <Form.Item
              name="auto_retrain"
              label="Auto-retrain when new data is available"
              valuePropName="checked"
            >
              <Switch />
            </Form.Item>

            <Form.Item
              name="retrain_threshold"
              label="Auto-retrain threshold (new examples)"
            >
              <Input type="number" />
            </Form.Item>

            <Form.Item>
              <Button type="primary" htmlType="submit">
                Save Changes
              </Button>
            </Form.Item>
          </Form>
        </Card>
      ),
    },
    {
      key: "notifications",
      label: "Notifications",
      children: (
        <Card>
          <Form layout="vertical" style={{ maxWidth: 600 }}>
            <Title level={5}>Email Notifications</Title>
            <Form.Item label="Detection alerts" valuePropName="checked">
              <Switch defaultChecked />
            </Form.Item>
            <Form.Item label="Weekly summary report" valuePropName="checked">
              <Switch defaultChecked />
            </Form.Item>
            <Form.Item
              label="Model retrain completed"
              valuePropName="checked"
            >
              <Switch defaultChecked />
            </Form.Item>
            <Form.Item
              label="Pending rules threshold reached"
              valuePropName="checked"
            >
              <Switch defaultChecked />
            </Form.Item>
            <Form.Item
              label="API key expiration warning"
              valuePropName="checked"
            >
              <Switch defaultChecked />
            </Form.Item>

            <Divider />

            <Title level={5}>In-App Notifications</Title>
            <Form.Item
              label="Show browser notifications"
              valuePropName="checked"
            >
              <Switch />
            </Form.Item>

            <Form.Item>
              <Button type="primary">Save Preferences</Button>
            </Form.Item>
          </Form>
        </Card>
      ),
    },
    {
      key: "security",
      label: "Security",
      children: (
        <Card>
          <Form layout="vertical" style={{ maxWidth: 600 }}>
            <Title level={5}>Authentication</Title>
            <Form.Item
              label="Enforce two-factor authentication"
              valuePropName="checked"
            >
              <Switch />
            </Form.Item>
            <Form.Item label="Session timeout (minutes)">
              <Input type="number" defaultValue={60} />
            </Form.Item>
            <Form.Item label="Maximum login attempts">
              <Input type="number" defaultValue={5} />
            </Form.Item>

            <Divider />

            <Title level={5}>Password Policy</Title>
            <Form.Item label="Minimum password length">
              <Input type="number" defaultValue={8} />
            </Form.Item>
            <Form.Item
              label="Require special characters"
              valuePropName="checked"
            >
              <Switch defaultChecked />
            </Form.Item>
            <Form.Item
              label="Require numbers"
              valuePropName="checked"
            >
              <Switch defaultChecked />
            </Form.Item>

            <Divider />

            <Title level={5}>IP Allowlist</Title>
            <Form.Item
              label="Allowed IP addresses (one per line)"
              extra="Leave empty to allow all"
            >
              <Input.TextArea rows={4} placeholder="192.168.1.0/24" />
            </Form.Item>

            <Form.Item>
              <Button type="primary">Save Security Settings</Button>
            </Form.Item>
          </Form>
        </Card>
      ),
    },
    {
      key: "import-export",
      label: "Import/Export",
      children: (
        <Card>
          <Space direction="vertical" size="large" style={{ width: "100%" }}>
            <div>
              <Title level={5}>Export Configuration</Title>
              <Text type="secondary">
                Export all PII types, rules, and configurations as JSON.
              </Text>
              <div style={{ marginTop: 12 }}>
                <Button icon={<DownloadOutlined />} type="primary">
                  Export All Configuration
                </Button>
              </div>
            </div>

            <Divider />

            <div>
              <Title level={5}>Import Configuration</Title>
              <Text type="secondary">
                Import PII types, rules, and configurations from a JSON file.
              </Text>
              <div style={{ marginTop: 12 }}>
                <Upload
                  accept=".json"
                  maxCount={1}
                  beforeUpload={() => false}
                >
                  <Button icon={<UploadOutlined />}>
                    Select JSON File
                  </Button>
                </Upload>
              </div>
            </div>

            <Divider />

            <div>
              <Title level={5}>Export Training Data</Title>
              <Text type="secondary">
                Export training data for use with external ML tools.
              </Text>
              <div style={{ marginTop: 12 }}>
                <Space>
                  <Button icon={<DownloadOutlined />}>
                    Export as JSON
                  </Button>
                  <Button icon={<DownloadOutlined />}>
                    Export as CSV
                  </Button>
                </Space>
              </div>
            </div>
          </Space>
        </Card>
      ),
    },
  ];

  return (
    <div>
      <Title level={4} style={{ marginBottom: 24 }}>
        Settings
      </Title>
      <Tabs
        activeKey={activeTab}
        onChange={handleTabChange}
        items={tabItems}
      />
    </div>
  );
};

export default SettingsPage;
