import React from "react";

type PerformanceMetricsProps = {
  epoch?: number;
  accuracy?: number;
  loss?: number;
  speed?: number;
  trainingTime?: number;
  theme: Theme;
};

type Theme = {
  controlBg: string;
  shadow: string;
  accent: string;
};

const PerformanceMetrics: React.FC<PerformanceMetricsProps> = ({
  epoch,
  accuracy,
  loss,
  speed,
  trainingTime,
  theme,
}) => {
  const metrics = [
    { label: "Epoch", value: epoch !== undefined ? epoch : "—", show: epoch !== undefined },
    { label: "Accuracy", value: accuracy !== undefined ? `${(accuracy * 100).toFixed(1)}%` : "—", show: accuracy !== undefined },
    { label: "Loss", value: loss !== undefined ? loss.toFixed(4) : "—", show: loss !== undefined },
    { label: "Speed", value: speed !== undefined ? `${speed.toFixed(2)}x` : "—", show: speed !== undefined },
    { label: "Time", value: trainingTime !== undefined ? `${trainingTime.toFixed(1)}s` : "—", show: trainingTime !== undefined },
  ].filter((m) => m.show);

  if (metrics.length === 0) return null;

  return (
    <div
      style={{
        position: "fixed",
        top: 86,
        left: 24,
        zIndex: 1200,
        background: theme.controlBg,
        backdropFilter: "blur(20px)",
        padding: "12px 16px",
        borderRadius: 12,
        boxShadow: `0 8px 32px ${theme.shadow}`,
        border: `1px solid ${theme.shadow}`,
        minWidth: 200,
      }}
    >
      <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 10, opacity: 0.7 }}>
        Performance Metrics
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
        {metrics.map(({ label, value }) => (
          <div
            key={label}
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              fontSize: 14,
            }}
          >
            <span style={{ opacity: 0.8 }}>{label}:</span>
            <span style={{ fontWeight: 600, fontFamily: "monospace", color: theme.accent }}>
              {value}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
};

export default PerformanceMetrics;
