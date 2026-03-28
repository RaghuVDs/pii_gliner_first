import React, { useEffect, useState, useCallback } from "react";
import {
  Card,
  Table,
  Typography,
  Tag,
  Space,
  Select,
  DatePicker,
  Button,
  Input,
  message,
} from "antd";
import {
  DownloadOutlined,
  SearchOutlined,
} from "@ant-design/icons";
import { auditApi, AuditLogFilters } from "@/api/audit";
import { AuditLog } from "@/types/models";
import { usePagination } from "@/hooks/usePagination";
import dayjs from "dayjs";

const { Title, Text } = Typography;
const { RangePicker } = DatePicker;

const ACTION_COLORS: Record<string, string> = {
  create: "green",
  update: "blue",
  delete: "red",
  login: "cyan",
  logout: "default",
  detect: "purple",
  redact: "orange",
  export: "gold",
};

const AuditPage: React.FC = () => {
  const [logs, setLogs] = useState<AuditLog[]>([]);
  const [loading, setLoading] = useState(true);
  const [filters, setFilters] = useState<AuditLogFilters>({});
  const [searchUser, setSearchUser] = useState("");
  const { page, pageSize, setTotal, paginationProps } = usePagination();

  const fetchLogs = useCallback(async () => {
    setLoading(true);
    try {
      const data = await auditApi.listLogs(filters, page, pageSize);
      setLogs(data.items);
      setTotal(data.total);
    } catch {
      message.error("Failed to load audit logs");
    } finally {
      setLoading(false);
    }
  }, [filters, page, pageSize, setTotal]);

  useEffect(() => {
    fetchLogs();
  }, [fetchLogs]);

  const handleExport = async () => {
    try {
      const blob = await auditApi.exportLogs(filters);
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `audit-logs-${dayjs().format("YYYY-MM-DD")}.csv`;
      a.click();
      window.URL.revokeObjectURL(url);
      message.success("Export started");
    } catch {
      message.error("Export failed");
    }
  };

  const handleDateChange = (
    dates: [dayjs.Dayjs, dayjs.Dayjs] | null
  ) => {
    setFilters({
      ...filters,
      date_from: dates?.[0]?.format("YYYY-MM-DD"),
      date_to: dates?.[1]?.format("YYYY-MM-DD"),
    });
  };

  const columns = [
    {
      title: "Action",
      dataIndex: "action",
      key: "action",
      render: (action: string) => (
        <Tag color={ACTION_COLORS[action] || "default"}>
          {action.toUpperCase()}
        </Tag>
      ),
    },
    {
      title: "User",
      key: "user",
      render: (_: unknown, record: AuditLog) => (
        <Text>{record.user_email}</Text>
      ),
    },
    {
      title: "Resource",
      key: "resource",
      render: (_: unknown, record: AuditLog) => (
        <Space direction="vertical" size={0}>
          <Text>{record.resource_type}</Text>
          {record.resource_id && (
            <Text type="secondary" style={{ fontSize: 11 }}>
              {record.resource_id}
            </Text>
          )}
        </Space>
      ),
    },
    {
      title: "Details",
      dataIndex: "details",
      key: "details",
      ellipsis: true,
      render: (details: Record<string, unknown>) => (
        <Text
          type="secondary"
          style={{ fontSize: 12 }}
          ellipsis={{ tooltip: JSON.stringify(details, null, 2) }}
        >
          {JSON.stringify(details)}
        </Text>
      ),
    },
    {
      title: "IP Address",
      dataIndex: "ip_address",
      key: "ip_address",
      width: 120,
    },
    {
      title: "Timestamp",
      dataIndex: "created_at",
      key: "created_at",
      width: 160,
      render: (date: string) =>
        dayjs(date).format("MMM DD, YYYY HH:mm:ss"),
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
          Audit Log
        </Title>
        <Button
          icon={<DownloadOutlined />}
          onClick={handleExport}
        >
          Export CSV
        </Button>
      </div>

      {/* Filters */}
      <Card style={{ marginBottom: 16 }}>
        <Space wrap size="middle">
          <Select
            placeholder="Action type"
            style={{ width: 160 }}
            value={filters.action}
            onChange={(v) => setFilters({ ...filters, action: v })}
            allowClear
            options={Object.keys(ACTION_COLORS).map((a) => ({
              label: a.toUpperCase(),
              value: a,
            }))}
          />
          <Input
            placeholder="Search by user"
            prefix={<SearchOutlined />}
            style={{ width: 200 }}
            value={searchUser}
            onChange={(e) => setSearchUser(e.target.value)}
            onPressEnter={() =>
              setFilters({ ...filters, user_id: searchUser || undefined })
            }
            allowClear
          />
          <Select
            placeholder="Resource type"
            style={{ width: 160 }}
            value={filters.resource_type}
            onChange={(v) => setFilters({ ...filters, resource_type: v })}
            allowClear
            options={[
              { label: "User", value: "user" },
              { label: "API Key", value: "api_key" },
              { label: "PII Type", value: "pii_type" },
              { label: "Rule", value: "rule" },
              { label: "Detection", value: "detection" },
              { label: "Model", value: "model" },
            ]}
          />
          <RangePicker onChange={handleDateChange} />
        </Space>
      </Card>

      {/* Log Table */}
      <Card>
        <Table
          columns={columns}
          dataSource={logs}
          rowKey="id"
          loading={loading}
          size="small"
          pagination={paginationProps}
          scroll={{ x: 900 }}
        />
      </Card>
    </div>
  );
};

export default AuditPage;
