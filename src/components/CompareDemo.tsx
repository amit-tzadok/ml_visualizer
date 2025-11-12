import React, { useState } from "react";
import PerceptronDemo from "./PerceptronDemo";
import KnnDemo from "./KnnDemo";
import MlpDemo from "./MlpDemo";

type Point = { x: number; y: number; label: string };

type CompareDemoProps = {
  leftType?: string;
  rightType?: string;
  theme?: ThemeShape;
  speedScale?: number;
};

type ThemeShape = {
  controlBg?: string;
  text?: string;
  accent?: string;
  shadow?: string;
};

const CompareDemo: React.FC<CompareDemoProps> = ({
  leftType = "linear",
  rightType = "poly",
  theme,
  speedScale = 1,
}) => {
  const gaussianRandom = () => {
    let u = 0,
      v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  };

  const generateLinearDataset = (count = 80, noise = 0.12) => {
    const pts: Point[] = [];
    const cxA = -0.5,
      cyA = -0.2;
    const cxB = 0.5,
      cyB = 0.2;
    for (let i = 0; i < count; i++) {
      const isA = i < count / 2;
      const cx = isA ? cxA : cxB;
      const cy = isA ? cyA : cyB;
      const x = Math.max(-1, Math.min(1, cx + gaussianRandom() * noise));
      const y = Math.max(-1, Math.min(1, cy + gaussianRandom() * noise));
      pts.push({ x, y, label: isA ? "A" : "B" });
    }
    return pts;
  };

  const generateNonlinearDataset = (count = 80, noise = 0.12) => {
    const pts: Point[] = [];
    for (let i = 0; i < count; i++) {
      const outer = i < count / 2;
      const angle = Math.random() * Math.PI * 2;
      const r = outer
        ? 0.6 + (Math.random() - 0.5) * noise
        : 0.28 + (Math.random() - 0.5) * noise;
      const x = Math.max(-1, Math.min(1, Math.cos(angle) * r));
      const y = Math.max(-1, Math.min(1, Math.sin(angle) * r));
      pts.push({ x, y, label: outer ? "A" : "B" });
    }
    return pts;
  };

  const [dataset, setDataset] = useState<Point[]>(() => {
    // Initialize with a solvable linear dataset
    return generateLinearDataset(80, 0.12);
  });

  const [resetToken, setResetToken] = useState<number>(0);

  const onDatasetChange = (updater: ((d: Point[]) => Point[]) | Point[]) => {
    if (typeof updater === "function") setDataset((d) => (updater as (d: Point[]) => Point[])(d));
    else setDataset(updater);
  };

  const [datasetSize, setDatasetSize] = useState<number>(80);
  const [noiseLevel, setNoiseLevel] = useState<string>("medium");
  const noiseMap: Record<string, number> = { low: 0.06, medium: 0.12, high: 0.22 };

  const renderDemo = (type: string, key: string) => {
    const commonProps: Record<string, unknown> = {
      dataset,
      onDatasetChange,
      resetToken,
      showInstructions: false,
      theme,
      speedScale,
    };
    // Use a unique key that includes resetToken to force remount when needed
    const uniqueKey = `${key}-${resetToken}`;
    if (type === "knn") return <KnnDemo key={uniqueKey} {...commonProps} />;
    if (type === "mlp") return <MlpDemo key={uniqueKey} {...commonProps} />;
    return <PerceptronDemo key={uniqueKey} classifierType={type} {...commonProps} />;
  };

  return (
    <div style={{ width: "100%", height: "100%", display: "flex", flexDirection: "column" }}>
      <div className="compare-controls">
        <div className="compare-control-groups">
          <div className="compare-control">
            <span style={{ marginBottom: 6, opacity: 0.9 }}>Size</span>
            <select
              className="compare-select"
              value={datasetSize}
              onChange={(e) => setDatasetSize(Number(e.target.value))}
            >
              <option value={40}>40</option>
              <option value={80}>80</option>
              <option value={160}>160</option>
            </select>
          </div>

          <div className="compare-control">
            <span style={{ marginBottom: 6, opacity: 0.9 }}>Noise</span>
            <select
              className="compare-select"
              value={noiseLevel}
              onChange={(e) => setNoiseLevel(e.target.value)}
            >
              <option value="low">Low</option>
              <option value="medium">Medium</option>
              <option value="high">High</option>
            </select>
          </div>
        </div>

        <div className="compare-button-group">
          <button
            className="compare-button"
            onClick={() => {
              setDataset(generateLinearDataset(datasetSize, noiseMap[noiseLevel]));
              setResetToken((t) => t + 1);
            }}
            title="Load a linear-separable test dataset"
          >
            Load linear
          </button>
          <button
            className="compare-button"
            onClick={() => {
              setDataset(generateNonlinearDataset(datasetSize, noiseMap[noiseLevel]));
              setResetToken((t) => t + 1);
            }}
            title="Load a nonlinear (concentric rings) test dataset"
          >
            Load nonlinear
          </button>
          <button
            className="compare-button"
            onClick={() => {
              setDataset([]);
              setResetToken((t) => t + 1);
            }}
            title="Clear dataset"
          >
            Clear
          </button>
        </div>
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: 16,
          padding: "12px 16px",
          width: "100%",
          flex: 1,
          minHeight: 500,
          maxHeight: "100%",
          boxSizing: "border-box",
        }}
      >
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            gap: 6,
            height: "100%",
          }}
        >
          <h3 style={{ textAlign: "center", margin: 0, fontSize: "16px", fontWeight: "600", flexShrink: 0 }}>{leftType.toUpperCase()}</h3>
          <div style={{ width: "100%", height: "calc(100% - 26px)", border: "1px solid #e2e8f0", borderRadius: "8px", overflow: "hidden", backgroundColor: "rgba(255,255,255,0.5)", position: "relative" }}>{renderDemo(leftType, "left")}</div>
        </div>

        <div
          style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            gap: 6,
            height: "100%",
          }}
        >
          <h3 style={{ textAlign: "center", margin: 0, fontSize: "16px", fontWeight: "600", flexShrink: 0 }}>{rightType.toUpperCase()}</h3>
          <div style={{ width: "100%", height: "calc(100% - 26px)", border: "1px solid #e2e8f0", borderRadius: "8px", overflow: "hidden", backgroundColor: "rgba(255,255,255,0.5)", position: "relative" }}>{renderDemo(rightType, "right")}</div>
        </div>
      </div>
    </div>
  );
};

export default CompareDemo;
