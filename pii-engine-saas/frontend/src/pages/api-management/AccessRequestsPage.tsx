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
  message,
} from "antd";
import {
  CheckOutlined,
  CloseOutlined,
  EyeOutlined,
} from "@ant-design/icons";
import { apiKeysApi, ReviewRequestData } from "@/api/apiKeys";
import { APIAccessRequest } from "@/types/models";
import { APIRequestStatus } from "@/types/enums";
import dayjs from "dayjs";

const { Title, Text, Paragraph } = Typography;
const { TextArea } = Input;

const AccessRequestsPage: React.FC = () => {
  const [requests, setRequests] = useState<APIAccessRequest[]>([]);
  const [loading, setLoading] = useState(true);
  const [filterStatus, setFilterStatus] = useState<
    APIRequestStatus | undefined
  >();
  const [detailModal, setDetailModal] = useState<APIAccessRequest | null>(
    null
  );
  const [reviewNote, setReviewNote] = useState("");

  const fetchRequests = useCallback(async () => {
    setLoading(true);
    try {
      const data = await apiKeysApi.listAccessRequests(filterStatus);
      setRequests(data);
    } catch {
      message.error("Failed to load access requests");
    } finally {
      setLoading(false);
    }
  }, [filterStatus]);

  useEffect(() => {
    fetchRequests();
  }, [fetchRequests]);

  const handleReview = async (
    requestId: string,
    status: APIRequestStatus.APPROVED | APIRequestStatus.DENIED
  ) => {
    const data: ReviewRequestData = {
      status,
      review_note: reviewNote || undefined,
    };
    try {
      await apiKeysApi.reviewAccessRequest(requestId, data);
      message.success(
        `Request ${status === APIRequestStatus.APPROVED ? "approved" : "denied"}`
      );
      setDetailModal(null);
      setReviewNote("");
      fetchRequests();
    } catch {
      message.error("Failed to process request");
    }
  };

  const statusColors: Record<string, string> = {
    pending: "orange",
    approved: "green",
    denied: "red",
    revoked: "default",
  };

  const columns = [
    {
      title: "Requester",
      key: "requester",
      render: (_: unknown, record: APIAccessRequest) => (
        <Space direction="vertical" size={0}>
          <Text strong>{record.user_name}</Text>
          <Text type="secondary" style={{ fontSize: 12 }}>
            {record.user_email}
          </Text>
        </Space>
      ),
    },
    {
      title: "Requested Scopes",
      dataIndex: "requested_scopes",
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
      title: "Status",
      dataIndex: "status",
      key: "status",
      render: (status: string) => (
        <Tag color={statusColors[status] || "default"}>
          {status.toUpperCase()}
        </Tag>
      ),
    },
    {
      title: "Requested",
      dataIndex: "created_at",
      key: "created_at",
      render: (date: string) => dayjs(date).format("MMM DD, YYYY"),
    },
    {
      title: "Actions",
      key: "actions",
      render: (_: unknown, record: APIAccessRequest) => (
        <Space size="small">
          <Button
            type="text"
            icon={<EyeOutlined />}
            size="small"
            onClick={() => setDetailModal(record)}
          >
            View
          </Button>
          {record.status === APIRequestStatus.PENDING && (
            <>
              <Button
                type="text"
                icon={<CheckOutlined />}
                size="small"
                style={{ color: "#52c41a" }}
                onClick={() =>
                  handleReview(record.id, APIRequestStatus.APPROVED)
                }
              >
                Approve
              </Button>
              <Button
                type="text"
                icon={<CloseOutlined />}
                size="small"
                danger
                onClick={() =>
                  handleReview(record.id, APIRequestStatus.DENIED)
                }
              >
                Deny
              </Button>
            </>
          )}
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
          Access Requests
        </Title>
        <Select
          placeholder="Filter by status"
          style={{ width: 160 }}
          value={filterStatus}
          onChange={setFilterStatus}
          allowClear
          options={[
            { label: "Pending", value: APIRequestStatus.PENDING },
            { label: "Approved", value: APIRequestStatus.APPROVED },
            { label: "Denied", value: APIRequestStatus.DENIED },
          ]}
        />
      </div>

      <Card>
        <Table
          columns={columns}
          dataSource={requests}
          rowKey="id"
          loading={loading}
          size="small"
          pagination={{ pageSize: 20 }}
        />
      </Card>

      {/* Detail Modal */}
      <Modal
        title="Access Request Details"
        open={!!detailModal}
        onCancel={() => {
          setDetailModal(null);
          setReviewNote("");
        }}
        footer={
          detailModal?.status === APIRequestStatus.PENDING
            ? [
                <Button
                  key="deny"
                  danger
                  onClick={() =>
                    detailModal &&
                    handleReview(detailModal.id, APIRequestStatus.DENIED)
                  }
                >
                  Deny
                </Button>,
                <Button
                  key="approve"
                  type="primary"
                  onClick={() =>
                    detailModal &&
                    handleReview(detailModal.id, APIRequestStatus.APPROVED)
                  }
                >
                  Approve
                </Button>,
              ]
            : [
                <Button
                  key="close"
                  onClick={() => setDetailModal(null)}
                >
                  Close
                </Button>,
              ]
        }
        width={600}
      >
        {detailModal && (
          <Space direction="vertical" style={{ width: "100%" }} size="middle">
            <div>
              <Text strong>Requester: </Text>
              <Text>
                {detailModal.user_name} ({detailModal.user_email})
              </Text>
            </div>
            <div>
              <Text strong>Use Case:</Text>
              <Paragraph
                style={{
                  background: "#fafafa",
                  padding: 12,
                  borderRadius: 6,
                  marginTop: 4,
                }}
              >
                {detailModal.use_case}
              </Paragraph>
            </div>
            <div>
              <Text strong>Requested Scopes:</Text>
              <div style={{ marginTop: 8 }}>
                {detailModal.requested_scopes.map((s) => (
                  <Tag key={s} style={{ margin: 4 }}>
                    {s}
                  </Tag>
                ))}
              </div>
            </div>
            {detailModal.status === APIRequestStatus.PENDING && (
              <Form.Item label="Review Note (optional)">
                <TextArea
                  rows={3}
                  value={reviewNote}
                  onChange={(e) => setReviewNote(e.target.value)}
                  placeholder="Add a note for the requester..."
                />
              </Form.Item>
            )}
            {detailModal.review_note && (
              <div>
                <Text strong>Review Note:</Text>
                <Paragraph>{detailModal.review_note}</Paragraph>
              </div>
            )}
          </Space>
        )}
      </Modal>
    </div>
  );
};

export default AccessRequestsPage;
