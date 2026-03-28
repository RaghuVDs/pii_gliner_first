import React, { useEffect } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import {
  Layout,
  Breadcrumb,
  Badge,
  Dropdown,
  Avatar,
  Space,
  Input,
  Typography,
} from "antd";
import type { MenuProps } from "antd";
import {
  BellOutlined,
  UserOutlined,
  LogoutOutlined,
  SettingOutlined,
  SearchOutlined,
} from "@ant-design/icons";
import { useAuthStore } from "@/store/authStore";
import { useNotificationStore } from "@/store/notificationStore";
import { useAuth } from "@/hooks/useAuth";

const { Header: AntHeader } = Layout;
const { Text } = Typography;

const BREADCRUMB_MAP: Record<string, string> = {
  dashboard: "Dashboard",
  detection: "Detection",
  taxonomy: "PII Taxonomy",
  rules: "Rules",
  regex: "Regex Rules",
  "field-patterns": "Field Patterns",
  context: "Context Rules",
  masking: "Masking Rules",
  learning: "Learning",
  "pending-rules": "Pending Rules",
  "training-data": "Training Data",
  models: "ML Models",
  api: "API Management",
  keys: "API Keys",
  requests: "Access Requests",
  usage: "Usage Analytics",
  users: "Users",
  audit: "Audit Log",
  settings: "Settings",
  profile: "Profile",
};

const Header: React.FC = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const user = useAuthStore((state) => state.user);
  const { unreadCount, fetchNotifications } = useNotificationStore();
  const { logout } = useAuth();

  const isAuthenticated = useAuthStore((s) => s.isAuthenticated);
  useEffect(() => {
    if (isAuthenticated) {
      fetchNotifications();
    }
  }, [fetchNotifications, isAuthenticated]);

  const pathSegments = location.pathname.split("/").filter(Boolean);
  const breadcrumbItems = pathSegments.map((segment, index) => {
    const path = "/" + pathSegments.slice(0, index + 1).join("/");
    return {
      title: BREADCRUMB_MAP[segment] || segment,
      href: index < pathSegments.length - 1 ? path : undefined,
    };
  });

  const userMenuItems: MenuProps["items"] = [
    {
      key: "profile",
      icon: <UserOutlined />,
      label: "Profile",
      onClick: () => navigate("/profile"),
    },
    {
      key: "settings",
      icon: <SettingOutlined />,
      label: "Settings",
      onClick: () => navigate("/settings"),
    },
    { type: "divider" },
    {
      key: "logout",
      icon: <LogoutOutlined />,
      label: "Logout",
      onClick: logout,
    },
  ];

  return (
    <AntHeader
      style={{
        background: "#fff",
        padding: "0 24px",
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        borderBottom: "1px solid #f0f0f0",
        position: "sticky",
        top: 0,
        zIndex: 10,
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
        <Breadcrumb items={breadcrumbItems} />
      </div>

      <Space size={20} align="center">
        <Input
          placeholder="Search..."
          prefix={<SearchOutlined />}
          style={{ width: 240 }}
          allowClear
        />

        <Badge count={unreadCount} size="small">
          <BellOutlined
            style={{ fontSize: 18, cursor: "pointer" }}
            onClick={() => navigate("/settings/notifications")}
          />
        </Badge>

        <Dropdown menu={{ items: userMenuItems }} trigger={["click"]}>
          <Space style={{ cursor: "pointer" }}>
            <Avatar
              size="small"
              icon={<UserOutlined />}
              style={{ backgroundColor: "#1677ff" }}
            />
            <Text strong>{user?.full_name || "User"}</Text>
          </Space>
        </Dropdown>
      </Space>
    </AntHeader>
  );
};

export default Header;
