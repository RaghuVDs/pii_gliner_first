import React from "react";
import { Tag } from "antd";
import { DetectionSource } from "@/types/enums";

const SOURCE_CONFIG: Record<
  DetectionSource,
  { color: string; label: string }
> = {
  [DetectionSource.GLINER]: { color: "blue", label: "GLiNER" },
  [DetectionSource.REGEX]: { color: "green", label: "Regex" },
  [DetectionSource.CONTEXT]: { color: "orange", label: "Context" },
  [DetectionSource.LSTM]: { color: "purple", label: "LSTM" },
  [DetectionSource.FIELD_PATTERN]: { color: "cyan", label: "Field Pattern" },
};

interface SourceBadgeProps {
  source: DetectionSource;
}

const SourceBadge: React.FC<SourceBadgeProps> = ({ source }) => {
  const config = SOURCE_CONFIG[source] || { color: "default", label: source };

  return <Tag color={config.color}>{config.label}</Tag>;
};

export default SourceBadge;
