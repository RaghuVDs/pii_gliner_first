import React, { useState, useCallback } from "react";
import {
  Row,
  Col,
  Card,
  Input,
  Button,
  Space,
  Typography,
  Table,
  Tag,
  Select,
  Statistic,
  message,
  Tooltip,
} from "antd";
import {
  ScanOutlined,
  EyeInvisibleOutlined,
  CopyOutlined,
  ClearOutlined,
} from "@ant-design/icons";
import { detectionApi } from "@/api/detection";
import { Detection, DetectionStats } from "@/types/models";
import { DetectionSource } from "@/types/enums";
import SourceBadge from "@/components/common/SourceBadge";
import ScoreIndicator from "@/components/common/ScoreIndicator";

const { Title, Text, Paragraph } = Typography;
const { TextArea } = Input;

const PII_TYPE_OPTIONS = [
  "PERSON",
  "PERSON_FULL_NAME",
  "EMAIL",
  "PHONE_NUMBER",
  "SSN",
  "CREDIT_CARD",
  "ADDRESS",
  "DATE_OF_BIRTH",
  "IP_ADDRESS",
  "PASSPORT",
  "DRIVERS_LICENSE",
  "BANK_ACCOUNT",
  "MEDICAL_RECORD",
];

const DetectionPage: React.FC = () => {
  const [inputText, setInputText] = useState("");
  const [selectedTypes, setSelectedTypes] = useState<string[]>([]);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [stats, setStats] = useState<DetectionStats | null>(null);
  const [redactedText, setRedactedText] = useState("");
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState<"detect" | "redact">("detect");

  const handleDetect = useCallback(async () => {
    if (!inputText.trim()) {
      message.warning("Please enter some text to analyze");
      return;
    }
    setLoading(true);
    setActiveTab("detect");
    try {
      const result = await detectionApi.detect(
        inputText,
        selectedTypes.length > 0 ? selectedTypes : undefined
      );
      const resultDetections = result.detections || [];
      setDetections(resultDetections);
      setStats(result.stats || null);
      setRedactedText("");
      const count =
        result.stats?.total ??
        result.detection_count ??
        resultDetections.length;
      const timeInfo = result.processing_time_ms
        ? ` in ${result.processing_time_ms}ms`
        : "";
      message.success(`Found ${count} PII entities${timeInfo}`);
    } catch {
      message.error("Detection failed");
    } finally {
      setLoading(false);
    }
  }, [inputText, selectedTypes]);

  const handleRedact = useCallback(async () => {
    if (!inputText.trim()) {
      message.warning("Please enter some text to redact");
      return;
    }
    setLoading(true);
    setActiveTab("redact");
    try {
      const result = await detectionApi.redact(
        inputText,
        selectedTypes.length > 0 ? selectedTypes : undefined
      );
      const resultDetections = result.detections || [];
      setDetections(resultDetections);
      setStats(result.stats || null);
      setRedactedText(result.redacted_text || "");
      const count =
        result.stats?.total ??
        result.detection_count ??
        resultDetections.length;
      message.success(`Redacted ${count} PII entities`);
    } catch {
      message.error("Redaction failed");
    } finally {
      setLoading(false);
    }
  }, [inputText, selectedTypes]);

  const handleClear = () => {
    setInputText("");
    setDetections([]);
    setRedactedText("");
    setStats(null);
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    message.success("Copied to clipboard");
  };

  // Build annotated text with highlights
  const renderAnnotatedText = () => {
    if (!inputText || detections.length === 0) return null;

    const sorted = [...detections].sort((a, b) => a.start - b.start);
    const parts: React.ReactNode[] = [];
    let lastEnd = 0;

    sorted.forEach((det, idx) => {
      if (det.start > lastEnd) {
        parts.push(
          <span key={`text-${idx}`}>
            {inputText.slice(lastEnd, det.start)}
          </span>
        );
      }
      parts.push(
        <Tooltip
          key={`det-${idx}`}
          title={`${det.label} (${(det.score * 100).toFixed(1)}%) - ${det.source}`}
        >
          <span className={`pii-highlight pii-highlight-default`}>
            {inputText.slice(det.start, det.end)}
            <Tag
              color="blue"
              style={{
                fontSize: 10,
                marginLeft: 4,
                verticalAlign: "super",
                lineHeight: 1,
                padding: "0 3px",
              }}
            >
              {det.label}
            </Tag>
          </span>
        </Tooltip>
      );
      lastEnd = det.end;
    });

    if (lastEnd < inputText.length) {
      parts.push(
        <span key="text-end">{inputText.slice(lastEnd)}</span>
      );
    }

    return (
      <div
        style={{
          padding: 16,
          background: "#fafafa",
          borderRadius: 8,
          lineHeight: 2,
          whiteSpace: "pre-wrap",
        }}
      >
        {parts}
      </div>
    );
  };

  const detectionColumns = [
    {
      title: "Label",
      dataIndex: "label",
      key: "label",
      render: (label: string) => (
        <Tag color="blue">{label}</Tag>
      ),
    },
    {
      title: "Text",
      dataIndex: "text",
      key: "text",
      render: (text: string) => (
        <Text code style={{ fontSize: 13 }}>
          {text}
        </Text>
      ),
    },
    {
      title: "Confidence",
      dataIndex: "score",
      key: "score",
      width: 160,
      render: (score: number) => (
        <ScoreIndicator score={score} size="small" />
      ),
    },
    {
      title: "Source",
      dataIndex: "source",
      key: "source",
      width: 120,
      render: (source: DetectionSource | string) => {
        // Handle string sources that may not match the enum exactly
        const enumVal = Object.values(DetectionSource).includes(
          source as DetectionSource
        )
          ? (source as DetectionSource)
          : undefined;
        return enumVal ? (
          <SourceBadge source={enumVal} />
        ) : (
          <Tag>{String(source)}</Tag>
        );
      },
    },
    {
      title: "Position",
      key: "position",
      width: 100,
      render: (_: unknown, record: Detection) => (
        <Text type="secondary" style={{ fontSize: 12 }}>
          [{record.start}:{record.end}]
        </Text>
      ),
    },
  ];

  return (
    <div>
      <Title level={4} style={{ marginBottom: 24 }}>
        PII Detection
      </Title>

      <Row gutter={[16, 16]}>
        {/* Input Panel */}
        <Col xs={24} lg={12}>
          <Card title="Input Text" style={{ height: "100%" }}>
            <Space direction="vertical" style={{ width: "100%" }} size="middle">
              <TextArea
                rows={10}
                placeholder="Enter text to scan for PII entities..."
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                style={{ fontFamily: "monospace" }}
              />

              <Select
                mode="multiple"
                placeholder="Filter PII types (optional - leave empty for all)"
                style={{ width: "100%" }}
                value={selectedTypes}
                onChange={setSelectedTypes}
                options={PII_TYPE_OPTIONS.map((t) => ({
                  label: t.replace(/_/g, " "),
                  value: t,
                }))}
                allowClear
              />

              <Space>
                <Button
                  type="primary"
                  icon={<ScanOutlined />}
                  onClick={handleDetect}
                  loading={loading && activeTab === "detect"}
                  size="large"
                >
                  Detect PII
                </Button>
                <Button
                  icon={<EyeInvisibleOutlined />}
                  onClick={handleRedact}
                  loading={loading && activeTab === "redact"}
                  size="large"
                >
                  Redact PII
                </Button>
                <Button
                  icon={<ClearOutlined />}
                  onClick={handleClear}
                >
                  Clear
                </Button>
              </Space>
            </Space>
          </Card>
        </Col>

        {/* Results Panel */}
        <Col xs={24} lg={12}>
          <Space direction="vertical" style={{ width: "100%" }} size={16}>
            {/* Stats Summary */}
            {stats && (
              <Card size="small">
                <Row gutter={16}>
                  <Col span={8}>
                    <Statistic
                      title="Total Detections"
                      value={stats.total}
                    />
                  </Col>
                  <Col span={8}>
                    <Statistic
                      title="PII Types"
                      value={Object.keys(stats.by_type || {}).length}
                    />
                  </Col>
                  <Col span={8}>
                    <Statistic
                      title="Sources"
                      value={Object.keys(stats.by_source || {}).length}
                    />
                  </Col>
                </Row>
                {stats.by_type && Object.keys(stats.by_type).length > 0 && (
                  <div style={{ marginTop: 12 }}>
                    {Object.entries(stats.by_type).map(([type, count]) => (
                      <Tag key={type} color="blue" style={{ marginBottom: 4 }}>
                        {type}: {count}
                      </Tag>
                    ))}
                  </div>
                )}
              </Card>
            )}

            {/* Annotated Text */}
            {detections.length > 0 && (
              <Card title="Annotated Text">
                {renderAnnotatedText()}
              </Card>
            )}

            {/* Redacted Text */}
            {redactedText && (
              <Card
                title="Redacted Text"
                extra={
                  <Button
                    type="text"
                    icon={<CopyOutlined />}
                    onClick={() => copyToClipboard(redactedText)}
                  >
                    Copy
                  </Button>
                }
              >
                <Paragraph
                  style={{
                    fontFamily: "monospace",
                    background: "#fafafa",
                    padding: 16,
                    borderRadius: 8,
                    whiteSpace: "pre-wrap",
                  }}
                >
                  {redactedText}
                </Paragraph>
              </Card>
            )}

            {/* Detection Table */}
            {detections.length > 0 && (
              <Card title={`Detections (${detections.length})`}>
                <Table
                  columns={detectionColumns}
                  dataSource={detections.map((d, i) => ({ ...d, key: i }))}
                  pagination={
                    detections.length > 10
                      ? { pageSize: 10, showSizeChanger: true }
                      : false
                  }
                  size="small"
                  scroll={{ x: 600 }}
                />
              </Card>
            )}
          </Space>
        </Col>
      </Row>
    </div>
  );
};

export default DetectionPage;
