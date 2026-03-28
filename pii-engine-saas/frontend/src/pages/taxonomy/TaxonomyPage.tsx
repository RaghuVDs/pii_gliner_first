import React, { useEffect, useState, useCallback } from "react";
import {
  Card,
  Input,
  Select,
  Switch,
  Slider,
  Button,
  Typography,
  Space,
  Row,
  Col,
  Collapse,
  Tag,
  Modal,
  Form,
  message,
  Spin,
  Tooltip,
  Drawer,
  Table,
  Tabs,
  Badge,
  Empty,
  Popconfirm,
} from "antd";
import {
  SearchOutlined,
  PlusOutlined,
  ReloadOutlined,
  EditOutlined,
  DeleteOutlined,
  EyeOutlined,
  CodeOutlined,
  TagOutlined,
  LockOutlined,
  SafetyOutlined,
} from "@ant-design/icons";
import { piiTypesApi } from "@/api/piiTypes";
import { rulesApi } from "@/api/rules";
import { PIICategory, PIIType, CustomPIIType } from "@/types/models";

const { Title, Text, Paragraph } = Typography;
const { Panel } = Collapse;
const { TabPane } = Tabs;

const CATEGORY_COLORS: Record<string, string> = {
  general: "#1677ff",
  country_specific: "#fa541c",
  financial_general: "#faad14",
  amex_specific: "#722ed1",
  medical: "#f5222d",
  education: "#13c2c2",
  employment: "#2f54eb",
  digital_security: "#a0d911",
  vehicle_property: "#fa8c16",
  legal_criminal: "#eb2f96",
  communication_utility: "#52c41a",
  contextual: "#ff4d4f",
  fallback: "#8c8c8c",
};

const TIER_LABELS: Record<number, { label: string; color: string }> = {
  0: { label: "Standard", color: "blue" },
  1: { label: "Sensitive", color: "orange" },
  2: { label: "Highly Sensitive", color: "red" },
  3: { label: "Contextual", color: "purple" },
};

interface RegexRule {
  id: number;
  pii_type_name: string;
  pattern: string;
  description?: string;
  is_system?: boolean;
  is_enabled?: boolean;
}

interface FieldPattern {
  id: number;
  pii_type_name: string;
  label_pattern: string;
  is_system?: boolean;
}

interface ContextRule {
  id: number;
  pii_type_name: string;
  keyword_pattern: string;
  is_negative?: boolean;
  is_system?: boolean;
}

const TaxonomyPage: React.FC = () => {
  const [categories, setCategories] = useState<PIICategory[]>([]);
  const [piiTypes, setPiiTypes] = useState<PIIType[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");
  const [filterCategory, setFilterCategory] = useState<string | undefined>();
  const [filterTier, setFilterTier] = useState<number | undefined>();
  const [filterEnabled, setFilterEnabled] = useState<boolean | undefined>();
  const [addModalOpen, setAddModalOpen] = useState(false);
  const [addForm] = Form.useForm();

  // Drawer state for selected PII type detail
  const [selectedType, setSelectedType] = useState<PIIType | null>(null);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [regexRules, setRegexRules] = useState<RegexRule[]>([]);
  const [fieldPatterns, setFieldPatterns] = useState<FieldPattern[]>([]);
  const [contextRules, setContextRules] = useState<ContextRule[]>([]);
  const [rulesLoading, setRulesLoading] = useState(false);
  const [addRegexModalOpen, setAddRegexModalOpen] = useState(false);
  const [regexForm] = Form.useForm();

  const fetchData = useCallback(async () => {
    setLoading(true);
    try {
      const [cats, types] = await Promise.all([
        piiTypesApi.listCategories(),
        piiTypesApi.listPIITypes({ search, pageSize: 200 }),
      ]);
      setCategories(cats);
      setPiiTypes(types.items || types);
    } catch {
      message.error("Failed to load taxonomy data");
    } finally {
      setLoading(false);
    }
  }, [search]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  // Load rules for selected PII type
  const loadRulesForType = async (typeName: string) => {
    setRulesLoading(true);
    try {
      const [regex, fields, context] = await Promise.allSettled([
        rulesApi.listRegexRules(typeName),
        rulesApi.listFieldPatterns(typeName),
        rulesApi.listContextRules(typeName),
      ]);
      setRegexRules(
        regex.status === "fulfilled" ? (regex.value as RegexRule[]) : []
      );
      setFieldPatterns(
        fields.status === "fulfilled" ? (fields.value as FieldPattern[]) : []
      );
      setContextRules(
        context.status === "fulfilled" ? (context.value as ContextRule[]) : []
      );
    } catch {
      // silently fail
    } finally {
      setRulesLoading(false);
    }
  };

  const handleTypeClick = (piiType: PIIType) => {
    setSelectedType(piiType);
    setDrawerOpen(true);
    loadRulesForType(piiType.name);
  };

  const handleToggle = async (piiTypeId: number, enabled: boolean) => {
    try {
      await piiTypesApi.updateConfig(piiTypeId, { is_enabled: enabled });
      setPiiTypes((prev) =>
        prev.map((t) =>
          t.id === piiTypeId ? { ...t, is_enabled: enabled } : t
        )
      );
    } catch {
      message.error("Failed to update configuration");
    }
  };

  const handleThresholdChange = async (piiTypeId: number, threshold: number) => {
    try {
      await piiTypesApi.updateConfig(piiTypeId, { custom_threshold: threshold });
    } catch {
      message.error("Failed to update threshold");
    }
  };

  const handleAddRegex = async (values: any) => {
    if (!selectedType) return;
    try {
      await rulesApi.createRegexRule({
        pii_type_name: selectedType.name,
        pattern: values.pattern,
        description: values.description,
      });
      message.success("Regex rule added");
      setAddRegexModalOpen(false);
      regexForm.resetFields();
      loadRulesForType(selectedType.name);
    } catch {
      message.error("Failed to add regex rule");
    }
  };

  const handleDeleteRegex = async (ruleId: number) => {
    try {
      await rulesApi.deleteRegexRule(String(ruleId));
      message.success("Rule deleted");
      if (selectedType) loadRulesForType(selectedType.name);
    } catch {
      message.error("Failed to delete rule");
    }
  };

  const handleAddCustomType = async (values: any) => {
    try {
      const aliases = values.gliner_aliases
        ? values.gliner_aliases.split(",").map((s: string) => s.trim())
        : [];
      const tags = values.compliance_tags
        ? values.compliance_tags.split(",").map((s: string) => s.trim())
        : [];
      await piiTypesApi.createCustomType({
        ...values,
        gliner_aliases: aliases,
        compliance_tags: tags,
      });
      message.success("Custom PII type created");
      setAddModalOpen(false);
      addForm.resetFields();
      fetchData();
    } catch {
      message.error("Failed to create custom PII type");
    }
  };

  const handleResetDefaults = () => {
    Modal.confirm({
      title: "Reset to Defaults",
      content: "This will reset all PII type configurations to system defaults.",
      onOk: async () => {
        try {
          await piiTypesApi.resetDefaults();
          message.success("Reset to defaults");
          fetchData();
        } catch {
          message.error("Reset failed");
        }
      },
    });
  };

  // Filter and group
  const filtered = piiTypes.filter((t) => {
    if (filterCategory && t.category_name !== filterCategory && String(t.category_id) !== filterCategory) return false;
    if (filterTier !== undefined && t.tier !== filterTier) return false;
    if (filterEnabled !== undefined && t.is_enabled !== filterEnabled) return false;
    if (search && !t.display_name.toLowerCase().includes(search.toLowerCase()) && !t.name.toLowerCase().includes(search.toLowerCase())) return false;
    return true;
  });

  const typesByCategory = categories.map((cat) => ({
    ...cat,
    types: filtered.filter(
      (t) => t.category_id === cat.id || t.category_name === cat.name
    ),
  })).filter(cat => cat.types.length > 0 || !search);

  if (loading) {
    return (
      <div style={{ textAlign: "center", padding: 80 }}>
        <Spin size="large" />
      </div>
    );
  }

  return (
    <div>
      {/* Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 24 }}>
        <div>
          <Title level={4} style={{ margin: 0 }}>PII Taxonomy & Rules</Title>
          <Text type="secondary">{piiTypes.length} PII types across {categories.length} categories. Click any type to view/edit its rules.</Text>
        </div>
        <Space>
          <Button icon={<ReloadOutlined />} onClick={handleResetDefaults}>Reset Defaults</Button>
          <Button type="primary" icon={<PlusOutlined />} onClick={() => setAddModalOpen(true)}>Add Custom PII Type</Button>
        </Space>
      </div>

      {/* Filters */}
      <Card size="small" style={{ marginBottom: 16 }}>
        <Row gutter={12} align="middle">
          <Col xs={24} sm={8}>
            <Input placeholder="Search PII types..." prefix={<SearchOutlined />} value={search} onChange={(e) => setSearch(e.target.value)} allowClear />
          </Col>
          <Col xs={8} sm={5}>
            <Select placeholder="Category" style={{ width: "100%" }} value={filterCategory} onChange={setFilterCategory} allowClear
              options={categories.map((c) => ({ label: c.display_name, value: c.name }))} />
          </Col>
          <Col xs={8} sm={4}>
            <Select placeholder="Tier" style={{ width: "100%" }} value={filterTier} onChange={setFilterTier} allowClear
              options={[
                { label: "Tier 0 - Standard", value: 0 },
                { label: "Tier 1 - Financial", value: 1 },
                { label: "Tier 2 - Sensitive", value: 2 },
                { label: "Tier 3 - Contextual", value: 3 },
              ]} />
          </Col>
          <Col xs={8} sm={4}>
            <Select placeholder="Status" style={{ width: "100%" }} value={filterEnabled} onChange={setFilterEnabled} allowClear
              options={[{ label: "Enabled", value: true }, { label: "Disabled", value: false }]} />
          </Col>
          <Col sm={3}>
            <Text type="secondary" style={{ fontSize: 12 }}>{filtered.length} of {piiTypes.length} shown</Text>
          </Col>
        </Row>
      </Card>

      {/* Category Sections */}
      <Collapse defaultActiveKey={categories.slice(0, 3).map((c) => c.name)} style={{ background: "#fff" }}>
        {typesByCategory.map((cat) => (
          <Panel
            key={cat.name}
            header={
              <Space>
                <Tag color={CATEGORY_COLORS[cat.name] || "default"} style={{ fontWeight: 600 }}>{cat.display_name}</Tag>
                <Badge count={cat.types.length} style={{ backgroundColor: cat.types.length > 0 ? "#1677ff" : "#d9d9d9" }} />
                {cat.tier !== undefined && (
                  <Tag color={TIER_LABELS[cat.tier]?.color || "default"} style={{ fontSize: 11 }}>
                    Tier {cat.tier}
                  </Tag>
                )}
              </Space>
            }
          >
            {cat.types.length === 0 ? (
              <Empty description="No PII types match filters" image={Empty.PRESENTED_IMAGE_SIMPLE} />
            ) : (
              <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
                {cat.types.map((piiType) => (
                  <Card
                    key={piiType.id}
                    size="small"
                    hoverable
                    onClick={() => handleTypeClick(piiType)}
                    style={{
                      width: 280,
                      borderLeft: `3px solid ${piiType.is_enabled ? (CATEGORY_COLORS[cat.name] || "#1677ff") : "#d9d9d9"}`,
                      opacity: piiType.is_enabled ? 1 : 0.6,
                      cursor: "pointer",
                    }}
                  >
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
                      <div style={{ flex: 1 }}>
                        <Text strong style={{ fontSize: 13 }}>{piiType.display_name}</Text>
                        <br />
                        <Text type="secondary" style={{ fontSize: 11 }}>{piiType.name}</Text>
                      </div>
                      <Switch
                        size="small"
                        checked={piiType.is_enabled}
                        onClick={(_, e) => e.stopPropagation()}
                        onChange={(checked) => handleToggle(piiType.id, checked)}
                      />
                    </div>
                    <div style={{ marginTop: 6, display: "flex", gap: 4, flexWrap: "wrap" }}>
                      <Tag style={{ fontSize: 10 }}>T: {piiType.default_threshold}</Tag>
                      {piiType.compliance_tags?.map((tag) => (
                        <Tag key={tag} color="blue" style={{ fontSize: 10 }}>{tag}</Tag>
                      ))}
                      {piiType.is_sensitive && <Tag color="red" style={{ fontSize: 10 }}>SENSITIVE</Tag>}
                    </div>
                  </Card>
                ))}
              </div>
            )}
          </Panel>
        ))}
      </Collapse>

      {/* Detail Drawer - shows rules for selected PII type */}
      <Drawer
        title={
          selectedType ? (
            <Space>
              <SafetyOutlined />
              <span>{selectedType.display_name}</span>
              <Tag>{selectedType.name}</Tag>
            </Space>
          ) : "PII Type Details"
        }
        open={drawerOpen}
        onClose={() => { setDrawerOpen(false); setSelectedType(null); }}
        width={640}
        destroyOnClose
      >
        {selectedType && (
          <div>
            {/* Overview */}
            <Card size="small" style={{ marginBottom: 16 }}>
              <Row gutter={16}>
                <Col span={12}>
                  <Text type="secondary">Status</Text>
                  <br />
                  <Switch checked={selectedType.is_enabled} onChange={(v) => handleToggle(selectedType.id, v)} />
                  <Text style={{ marginLeft: 8 }}>{selectedType.is_enabled ? "Enabled" : "Disabled"}</Text>
                </Col>
                <Col span={12}>
                  <Text type="secondary">Threshold</Text>
                  <Slider min={0} max={1} step={0.05} value={selectedType.default_threshold}
                    onChangeComplete={(v) => handleThresholdChange(selectedType.id, v)} />
                </Col>
              </Row>
              {selectedType.gliner_aliases && selectedType.gliner_aliases.length > 0 && (
                <div style={{ marginTop: 8 }}>
                  <Text type="secondary" style={{ fontSize: 12 }}>GLiNER Aliases:</Text>
                  <div style={{ marginTop: 4 }}>
                    {selectedType.gliner_aliases.map((a) => (
                      <Tag key={a} style={{ marginBottom: 4 }}>{a}</Tag>
                    ))}
                  </div>
                </div>
              )}
              {selectedType.compliance_tags && selectedType.compliance_tags.length > 0 && (
                <div style={{ marginTop: 8 }}>
                  <Text type="secondary" style={{ fontSize: 12 }}>Compliance:</Text>{" "}
                  {selectedType.compliance_tags.map((t) => (
                    <Tag key={t} color="blue">{t}</Tag>
                  ))}
                </div>
              )}
            </Card>

            {/* Rules Tabs */}
            <Spin spinning={rulesLoading}>
              <Tabs defaultActiveKey="regex" type="card">
                {/* Regex Rules Tab */}
                <TabPane
                  tab={<span><CodeOutlined /> Regex Rules ({regexRules.length})</span>}
                  key="regex"
                >
                  <div style={{ marginBottom: 8 }}>
                    <Button size="small" type="primary" icon={<PlusOutlined />} onClick={() => setAddRegexModalOpen(true)}>
                      Add Pattern
                    </Button>
                  </div>
                  {regexRules.length === 0 ? (
                    <Empty description="No regex rules" image={Empty.PRESENTED_IMAGE_SIMPLE} />
                  ) : (
                    <Table
                      dataSource={regexRules}
                      rowKey="id"
                      size="small"
                      pagination={false}
                      columns={[
                        {
                          title: "Pattern",
                          dataIndex: "pattern",
                          render: (p: string) => (
                            <Tooltip title={p}>
                              <code style={{ fontSize: 11, maxWidth: 350, display: "block", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{p}</code>
                            </Tooltip>
                          ),
                        },
                        {
                          title: "Source",
                          width: 80,
                          render: (_: any, r: RegexRule) => (
                            <Tag color={r.is_system ? "default" : "green"} style={{ fontSize: 10 }}>
                              {r.is_system ? "System" : "Custom"}
                            </Tag>
                          ),
                        },
                        {
                          title: "",
                          width: 40,
                          render: (_: any, r: RegexRule) => !r.is_system && (
                            <Popconfirm title="Delete this rule?" onConfirm={() => handleDeleteRegex(r.id)}>
                              <Button size="small" type="text" danger icon={<DeleteOutlined />} />
                            </Popconfirm>
                          ),
                        },
                      ]}
                    />
                  )}
                </TabPane>

                {/* Field Patterns Tab */}
                <TabPane
                  tab={<span><TagOutlined /> Field Patterns ({fieldPatterns.length})</span>}
                  key="field"
                >
                  {fieldPatterns.length === 0 ? (
                    <Empty description="No field patterns" image={Empty.PRESENTED_IMAGE_SIMPLE} />
                  ) : (
                    <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                      {fieldPatterns.map((fp) => (
                        <Tag key={fp.id} color="cyan" style={{ marginBottom: 4 }}>
                          {fp.label_pattern}
                        </Tag>
                      ))}
                    </div>
                  )}
                </TabPane>

                {/* Context Rules Tab */}
                <TabPane
                  tab={<span><EyeOutlined /> Context Rules ({contextRules.length})</span>}
                  key="context"
                >
                  {contextRules.length === 0 ? (
                    <Empty description="No context rules" image={Empty.PRESENTED_IMAGE_SIMPLE} />
                  ) : (
                    <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                      {contextRules.map((cr) => (
                        <Tag key={cr.id} color={cr.is_negative ? "red" : "green"} style={{ marginBottom: 4 }}>
                          {cr.is_negative ? "NEG: " : ""}{cr.keyword_pattern}
                        </Tag>
                      ))}
                    </div>
                  )}
                </TabPane>
              </Tabs>
            </Spin>
          </div>
        )}
      </Drawer>

      {/* Add Regex Rule Modal */}
      <Modal
        title={`Add Regex Rule for ${selectedType?.display_name || ""}`}
        open={addRegexModalOpen}
        onCancel={() => setAddRegexModalOpen(false)}
        onOk={() => regexForm.submit()}
      >
        <Form form={regexForm} layout="vertical" onFinish={handleAddRegex}>
          <Form.Item name="pattern" label="Regex Pattern" rules={[{ required: true }]}>
            <Input.TextArea rows={3} placeholder="e.g. (?i)\b(\d{3}-\d{2}-\d{4})\b" style={{ fontFamily: "monospace" }} />
          </Form.Item>
          <Form.Item name="description" label="Description">
            <Input placeholder="What does this pattern match?" />
          </Form.Item>
        </Form>
      </Modal>

      {/* Add Custom PII Type Modal */}
      <Modal title="Add Custom PII Type" open={addModalOpen} onCancel={() => setAddModalOpen(false)} onOk={() => addForm.submit()} width={600}>
        <Form form={addForm} layout="vertical" onFinish={handleAddCustomType}>
          <Form.Item name="name" label="Internal Name" rules={[{ required: true }]}>
            <Input placeholder="e.g. INTERNAL_PROJECT_CODE" />
          </Form.Item>
          <Form.Item name="display_name" label="Display Name" rules={[{ required: true }]}>
            <Input placeholder="e.g. Internal Project Code" />
          </Form.Item>
          <Form.Item name="category_id" label="Category">
            <Select options={categories.map((c) => ({ label: c.display_name, value: c.id }))} />
          </Form.Item>
          <Form.Item name="description" label="Description">
            <Input.TextArea rows={2} />
          </Form.Item>
          <Form.Item name="gliner_aliases" label="GLiNER Aliases (comma-separated)">
            <Input placeholder="project code, internal code" />
          </Form.Item>
          <Form.Item name="default_threshold" label="Threshold" initialValue={0.4}>
            <Slider min={0} max={1} step={0.05} />
          </Form.Item>
          <Form.Item name="compliance_tags" label="Compliance Tags (comma-separated)">
            <Input placeholder="GDPR, CCPA" />
          </Form.Item>
        </Form>
      </Modal>
    </div>
  );
};

export default TaxonomyPage;
