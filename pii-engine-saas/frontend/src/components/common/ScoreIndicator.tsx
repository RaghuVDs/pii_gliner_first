import React from "react";
import { Progress, Tooltip } from "antd";

interface ScoreIndicatorProps {
  score: number;
  showLabel?: boolean;
  size?: "small" | "default";
}

const getColor = (score: number): string => {
  if (score < 0.5) return "#ff4d4f";
  if (score < 0.7) return "#faad14";
  return "#52c41a";
};

const getLabel = (score: number): string => {
  if (score < 0.5) return "Low";
  if (score < 0.7) return "Medium";
  return "High";
};

const ScoreIndicator: React.FC<ScoreIndicatorProps> = ({
  score,
  showLabel = true,
  size = "default",
}) => {
  const color = getColor(score);
  const label = getLabel(score);
  const percent = Math.round(score * 100);

  return (
    <Tooltip title={`Confidence: ${(score * 100).toFixed(1)}%`}>
      <div
        style={{
          display: "inline-flex",
          alignItems: "center",
          gap: 8,
        }}
      >
        <Progress
          type="circle"
          percent={percent}
          size={size === "small" ? 32 : 44}
          strokeColor={color}
          format={() => `${percent}%`}
          strokeWidth={8}
        />
        {showLabel && (
          <span
            style={{
              color,
              fontWeight: 600,
              fontSize: size === "small" ? 12 : 14,
            }}
          >
            {label}
          </span>
        )}
      </div>
    </Tooltip>
  );
};

export default ScoreIndicator;
