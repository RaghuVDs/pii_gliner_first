import type { ThemeConfig } from "antd";

export const theme: ThemeConfig = {
  token: {
    colorPrimary: "#1677ff",
    colorSuccess: "#52c41a",
    colorWarning: "#faad14",
    colorError: "#ff4d4f",
    colorInfo: "#1677ff",
    borderRadius: 6,
    fontSize: 14,
    fontFamily:
      "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif",
    colorBgContainer: "#ffffff",
    colorBgLayout: "#f5f5f5",
  },
  components: {
    Layout: {
      siderBg: "#001529",
      headerBg: "#ffffff",
      bodyBg: "#f0f2f5",
    },
    Menu: {
      darkItemBg: "#001529",
      darkItemSelectedBg: "#1677ff",
    },
    Table: {
      headerBg: "#fafafa",
    },
    Card: {
      borderRadiusLG: 8,
    },
  },
};
