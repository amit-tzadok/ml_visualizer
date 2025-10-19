import React, { useState } from "react";
import PerceptronDemo from "./PerceptronDemo";
import KnnDemo from "./KnnDemo";
import MlpDemo from "./MlpDemo";

type Point = { x: number; y: number; label: string };

type CompareDemoProps = {
  leftType?: string;
  rightType?: string;
  theme?: any;
  speedScale?: number;
};

const CompareDemo: React.FC<CompareDemoProps> = ({
  leftType = "linear",
  rightType = "poly",
  theme,
  speedScale = 1,
}) => {
  const [dataset, setDataset] = useState<Point[]>(() => {
    const pts: Point[] = [];
    for (let i = 0; i < 40; i++) {
      const x = Math.random() * 2 - 1;
      const y = Math.random() * 2 - 1;
      const label = Math.random() > 0.5 ? "A" : "B";
      pts.push({ x, y, label });
    }
    return pts;
  });

  const [resetToken, setResetToken] = useState<number>(0);

  const onDatasetChange = (updater: any) => {
    if (typeof updater === "function") setDataset((d) => updater(d));
    else setDataset(updater);
  };

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

  const [datasetSize, setDatasetSize] = useState<number>(80);
  const [noiseLevel, setNoiseLevel] = useState<string>("medium");
  const noiseMap: Record<string, number> = { low: 0.06, medium: 0.12, high: 0.22 };

  const renderDemo = (type: string, key: string) => {
    const commonProps = {
      dataset,
      onDatasetChange,
      resetToken,
      showInstructions: false,
      theme,
      speedScale,
    } as any;
    if (type === "knn") return <KnnDemo key={key} {...commonProps} />;
    if (type === "mlp") return <MlpDemo key={key} {...commonProps} />;
    return <PerceptronDemo key={key} classifierType={type} {...commonProps} />;
  };

  return (
    <div>
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
          display: "flex",
          gap: 24,
          justifyContent: "center",
          alignItems: "flex-start",
          paddingTop: 24,
        }}
      >
        <div
          style={{
            width: 560,
            height: 560,
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            gap: 8,
          }}
        >
          <h3 style={{ textAlign: "center", margin: 0 }}>{leftType}</h3>
          <div style={{ width: "100%", height: "100%" }}>{renderDemo(leftType, "left")}</div>
        </div>

        <div
          style={{
            width: 560,
            height: 560,
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            gap: 8,
          }}
        >
          <h3 style={{ textAlign: "center", margin: 0 }}>{rightType}</h3>
          <div style={{ width: "100%", height: "100%" }}>{renderDemo(rightType, "right")}</div>
        </div>
      </div>
    </div>
  );
};

export default CompareDemo;
