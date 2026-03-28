import React from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { Layout, Menu } from "antd";
import type { MenuProps } from "antd";
import {
  DashboardOutlined,
  ScanOutlined,
  TagsOutlined,
  FileTextOutlined,
  BulbOutlined,
  RobotOutlined,
  KeyOutlined,
  TeamOutlined,
  AuditOutlined,
  SettingOutlined,
  FormOutlined,
  FieldStringOutlined,
  InteractionOutlined,
  EyeInvisibleOutlined,
  ExperimentOutlined,
  DatabaseOutlined,
  ApiOutlined,
  FileSearchOutlined,
  BarChartOutlined,
} from "@ant-design/icons";
import { useAuthStore } from "@/store/authStore";
import { UserRole } from "@/types/enums";

const { Sider } = Layout;

type MenuItem = Required<MenuProps>["items"][number];

interface SidebarProps {
  collapsed: boolean;
  onCollapse: (collapsed: boolean) => void;
}

const Sidebar: React.FC<SidebarProps> = ({ collapsed, onCollapse }) => {
  const navigate = useNavigate();
  const location = useLocation();
  const user = useAuthStore((state) => state.user);
  const isAdmin =
    user?.role === UserRole.TENANT_ADMIN ||
    user?.role === UserRole.SUPER_ADMIN;

  const menuItems: MenuItem[] = [
    {
      key: "/dashboard",
      icon: <DashboardOutlined />,
      label: "Dashboard",
    },
    {
      key: "/detection",
      icon: <ScanOutlined />,
      label: "Detection",
    },
    {
      key: "/taxonomy",
      icon: <TagsOutlined />,
      label: "PII Taxonomy & Rules",
    },
    {
      key: "learning",
      icon: <BulbOutlined />,
      label: "Learning",
      children: [
        {
          key: "/learning/pending-rules",
          icon: <ExperimentOutlined />,
          label: "Pending Rules",
        },
        {
          key: "/learning/training-data",
          icon: <DatabaseOutlined />,
          label: "Training Data",
        },
      ],
    },
    {
      key: "/models",
      icon: <RobotOutlined />,
      label: "ML Models",
    },
    {
      key: "api",
      icon: <ApiOutlined />,
      label: "API Management",
      children: [
        {
          key: "/api/keys",
          icon: <KeyOutlined />,
          label: "API Keys",
        },
        {
          key: "/api/requests",
          icon: <FileSearchOutlined />,
          label: "Access Requests",
        },
        {
          key: "/api/usage",
          icon: <BarChartOutlined />,
          label: "Usage Analytics",
        },
      ],
    },
    ...(isAdmin
      ? [
          {
            key: "/users",
            icon: <TeamOutlined />,
            label: "Users",
          } as MenuItem,
          {
            key: "/audit",
            icon: <AuditOutlined />,
            label: "Audit Log",
          } as MenuItem,
        ]
      : []),
    {
      key: "/settings",
      icon: <SettingOutlined />,
      label: "Settings",
    },
  ];

  const getSelectedKeys = (): string[] => {
    const path = location.pathname;
    return [path];
  };

  const getOpenKeys = (): string[] => {
    const path = location.pathname;
    if (path.startsWith("/rules")) return ["rules"];
    if (path.startsWith("/learning")) return ["learning"];
    if (path.startsWith("/api")) return ["api"];
    return [];
  };

  const handleMenuClick: MenuProps["onClick"] = (info) => {
    navigate(info.key);
  };

  return (
    <Sider
      collapsible
      collapsed={collapsed}
      onCollapse={onCollapse}
      width={240}
      style={{
        overflow: "auto",
        height: "100vh",
        position: "fixed",
        left: 0,
        top: 0,
        bottom: 0,
        zIndex: 100,
      }}
    >
      <div
        style={{
          height: 64,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          color: "#fff",
          fontSize: collapsed ? 16 : 20,
          fontWeight: 700,
          borderBottom: "1px solid rgba(255,255,255,0.1)",
        }}
      >
        {collapsed ? "PII" : "PII Engine"}
      </div>
      <Menu
        theme="dark"
        mode="inline"
        selectedKeys={getSelectedKeys()}
        defaultOpenKeys={getOpenKeys()}
        items={menuItems}
        onClick={handleMenuClick}
      />
    </Sider>
  );
};

export default Sidebar;
