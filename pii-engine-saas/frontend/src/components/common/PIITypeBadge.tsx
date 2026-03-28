import React from "react";
import { Tag } from "antd";
import { PIITier } from "@/types/enums";

const CATEGORY_COLORS: Record<string, string> = {
  personal_identifiers: "blue",
  financial: "gold",
  health: "red",
  contact: "green",
  government_id: "volcano",
  digital: "purple",
  biometric: "magenta",
  location: "cyan",
  employment: "geekblue",
  legal: "orange",
};

const TIER_COLORS: Record<PIITier, string> = {
  [PIITier.TIER_1]: "#ff4d4f",
  [PIITier.TIER_2]: "#faad14",
  [PIITier.TIER_3]: "#52c41a",
};

interface PIITypeBadgeProps {
  name: string;
  category?: string;
  tier?: PIITier;
  showTier?: boolean;
}

const PIITypeBadge: React.FC<PIITypeBadgeProps> = ({
  name,
  category,
  tier,
  showTier = false,
}) => {
  const color = category
    ? CATEGORY_COLORS[category] || "default"
    : "default";

  return (
    <span style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
      <Tag color={color}>{name.replace(/_/g, " ").toUpperCase()}</Tag>
      {showTier && tier && (
        <Tag
          color={TIER_COLORS[tier]}
          style={{ fontSize: 10, padding: "0 4px" }}
        >
          {tier.replace("_", " ").toUpperCase()}
        </Tag>
      )}
    </span>
  );
};

export default PIITypeBadge;
