/**
 * Logistic Regression interactive demo.
 * Uses the HTML Canvas 2D API (no p5.js) for rendering.
 * Animates gradient descent training with live decision boundary and loss curve.
 */
import React, { useRef, useEffect, useCallback, useState } from "react";
import LogisticRegression from "../utils/logisticRegression";
import {
  makeBlobs,
  makeMoons,
  makeCircles,
  makeXOR,
  makeSpirals,
  toXy,
  DataPoint,
} from "../utils/datasets";

type DatasetType = "blobs" | "moons" | "circles" | "xor" | "spirals";

interface ThemeColors {
  background: string;
  mainBackground: string;
  text: string;
  headerBg: string;
  controlBg: string;
  pointPositive: string;
  pointNegative: string;
  line: string;
  accent: string;
  shadow: string;
  textMuted?: string;
}

interface LogisticRegressionDemoProps {
  speedScale?: number;
  showInstructions?: boolean;
  theme?: ThemeColors;
}

// Canvas coordinate helpers — maps [-1.3, 1.3] <-> [0, size]
const RANGE = 1.3;
const toPx = (v: number, size: number) =>
  ((v + RANGE) / (2 * RANGE)) * size;
const fromPx = (px: number, size: number) =>
  (px / size) * (2 * RANGE) - RANGE;

const COLOR_A = [66, 153, 225] as const;   // class 1 — blue
const COLOR_B = [252, 129, 129] as const;  // class 0 — red
const MAX_EPOCHS = 500;

function blend(c1: readonly number[], c2: readonly number[], t: number) {
  return [
    Math.round(c1[0] * t + c2[0] * (1 - t)),
    Math.round(c1[1] * t + c2[1] * (1 - t)),
    Math.round(c1[2] * t + c2[2] * (1 - t)),
  ];
}

const LogisticRegressionDemo: React.FC<LogisticRegressionDemoProps> = ({
  speedScale = 1,
  theme,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const lossCanvasRef = useRef<HTMLCanvasElement>(null);
  const animRef = useRef<number | null>(null);
  const modelRef = useRef(new LogisticRegression(2, { lr: 0.3, lambda: 0.001 }));
  const ptsRef = useRef<DataPoint[]>([]);
  const lossHistoryRef = useRef<number[]>([]);
  const epochRef = useRef(0);
  const runningRef = useRef(false);
  const lastFrameTimeRef = useRef(0);

  const [datasetType, setDatasetType] = useState<DatasetType>("blobs");
  const [lr, setLr] = useState(0.3);
  const [isRunning, setIsRunning] = useState(false);
  const [epoch, setEpoch] = useState(0);
  const [currentLoss, setCurrentLoss] = useState<number | null>(null);
  const [accuracy, setAccuracy] = useState<number | null>(null);

  // Derived theme tokens — fall back to light-mode literals when no theme is provided
  const isDark = theme?.text === "#f7fafc";
  const T = {
    text:        theme?.text      ?? "#2d3748",
    subText:     theme?.textMuted ?? "#718096",
    labelText:   isDark ? "#cbd5e0" : "#4a5568",
    titleText:   theme?.text      ?? "#1a202c",
    controlBg:   theme?.controlBg ?? "rgba(255,255,255,0.98)",
    accent:      theme?.accent    ?? "#f43f5e",
    shadow:      theme?.shadow    ?? "rgba(0,0,0,0.08)",
    border:      `1px solid ${isDark ? "rgba(255,255,255,0.12)" : "#e2e8f0"}`,
    statsBg:     isDark ? "rgba(45,55,72,0.6)"    : "#f7fafc",
    infoBg:      isDark ? "rgba(43,108,176,0.15)" : "#ebf4ff",
    infoBorder:  isDark ? "1px solid rgba(102,126,234,0.3)" : "1px solid #bee3f8",
    infoText:    theme?.accent    ?? "#2b6cb0",
    btnBg:       theme?.controlBg ?? "#fff",
    btnText:     theme?.text      ?? "#4a5568",
    lossBg:      isDark ? "rgba(45,55,72,0.95)"   : "rgba(247,250,252,0.95)",
    lossText:    isDark ? "#a0aec0" : "#4a5568",
    lossPlaceholder: isDark ? "#718096" : "#a0aec0",
  };

  // Keep a ref to T so renderLoss (empty deps) can always read current theme
  const themeRef = useRef(T);
  themeRef.current = T;

  const generateDataset = useCallback((type: DatasetType): DataPoint[] => {
    switch (type) {
      case "moons":   return makeMoons(160, 0.1);
      case "circles": return makeCircles(160, 0.05, 0.38);
      case "xor":     return makeXOR(200, 0.08);
      case "spirals": return makeSpirals(200, 0.04);
      default:        return makeBlobs(160, 0.13);
    }
  }, []);

  const drawDecisionRegion = useCallback(
    (ctx: CanvasRenderingContext2D, model: LogisticRegression) => {
      const W = ctx.canvas.width;
      const H = ctx.canvas.height;
      const step = 3;
      const imageData = ctx.createImageData(W, H);
      const data = imageData.data;

      for (let px = 0; px < W; px += step) {
        for (let py = 0; py < H; py += step) {
          const x = fromPx(px, W);
          const y = fromPx(py, H);
          const prob = model.probability([x, y]);
          const [r, g, b] = blend(COLOR_A, COLOR_B, prob);

          for (let dx = 0; dx < step && px + dx < W; dx++) {
            for (let dy = 0; dy < step && py + dy < H; dy++) {
              const idx = ((py + dy) * W + (px + dx)) * 4;
              data[idx]     = r;
              data[idx + 1] = g;
              data[idx + 2] = b;
              data[idx + 3] = 70;
            }
          }
        }
      }
      ctx.putImageData(imageData, 0, 0);
    },
    []
  );

  const drawPoints = useCallback(
    (ctx: CanvasRenderingContext2D, pts: DataPoint[]) => {
      const W = ctx.canvas.width;
      const H = ctx.canvas.height;
      for (const pt of pts) {
        const px = toPx(pt.x, W);
        const py = toPx(pt.y, H);
        const [r, g, b] = pt.label === 1 ? COLOR_A : COLOR_B;
        ctx.beginPath();
        ctx.arc(px, py, 5, 0, Math.PI * 2);
        ctx.fillStyle = `rgb(${r},${g},${b})`;
        ctx.fill();
        ctx.strokeStyle = "rgba(255,255,255,0.75)";
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }
    },
    []
  );

  const drawDecisionLine = useCallback(
    (ctx: CanvasRenderingContext2D, model: LogisticRegression) => {
      const db = model.decisionBoundary2D();
      if (!db) return;
      const W = ctx.canvas.width;
      const H = ctx.canvas.height;
      const { slope, intercept } = db;
      const x0 = fromPx(0, W);
      const x1 = fromPx(W, W);
      const y0 = slope * x0 + intercept;
      const y1 = slope * x1 + intercept;

      ctx.beginPath();
      ctx.moveTo(toPx(x0, W), toPx(y0, H));
      ctx.lineTo(toPx(x1, W), toPx(y1, H));
      ctx.strokeStyle = "rgba(30,30,30,0.85)";
      ctx.lineWidth = 2.5;
      ctx.setLineDash([6, 4]);
      ctx.stroke();
      ctx.setLineDash([]);
    },
    []
  );

  const renderLoss = useCallback(() => {
    const canvas = lossCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const tc = themeRef.current;
    const W = canvas.width;
    const H = canvas.height;
    const losses = lossHistoryRef.current;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = tc.lossBg;
    ctx.fillRect(0, 0, W, H);

    if (losses.length < 2) {
      ctx.fillStyle = tc.lossPlaceholder;
      ctx.font = "12px system-ui";
      ctx.fillText("Start training to see loss curve", 16, H / 2 + 4);
      return;
    }

    const maxL = losses[0];
    const minL = Math.min(...losses);
    const range = Math.max(maxL - minL, 0.01);
    const pad = { l: 10, r: 10, t: 8, b: 10 };

    // Grid line at midpoint
    const midY = H - pad.b - ((maxL / 2 - minL) / range) * (H - pad.t - pad.b);
    ctx.strokeStyle = "rgba(203,213,224,0.5)";
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(pad.l, midY);
    ctx.lineTo(W - pad.r, midY);
    ctx.stroke();
    ctx.setLineDash([]);

    // Loss line
    ctx.beginPath();
    for (let i = 0; i < losses.length; i++) {
      const px =
        pad.l + (i / Math.max(losses.length - 1, 1)) * (W - pad.l - pad.r);
      const py =
        H - pad.b - ((losses[i] - minL) / range) * (H - pad.t - pad.b);
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }
    ctx.strokeStyle = tc.accent;
    ctx.lineWidth = 2;
    ctx.stroke();

    // Label
    ctx.fillStyle = tc.lossText;
    ctx.font = "bold 11px system-ui";
    ctx.fillText(
      `Loss: ${losses[losses.length - 1].toFixed(4)}`,
      pad.l + 2,
      14
    );
  }, []);

  const render = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    drawDecisionRegion(ctx, modelRef.current);
    drawPoints(ctx, ptsRef.current);
    drawDecisionLine(ctx, modelRef.current);
  }, [drawDecisionRegion, drawPoints, drawDecisionLine]);

  const doStep = useCallback(
    (currentLr: number) => {
      const { X, y } = toXy(ptsRef.current);
      const loss = modelRef.current.fitStep(X, y, currentLr);
      if (Number.isFinite(loss)) lossHistoryRef.current.push(loss);
      epochRef.current++;
      setEpoch(epochRef.current);
      setCurrentLoss(loss);
      setAccuracy(modelRef.current.accuracy(X, y));
      render();
      renderLoss();
    },
    [render, renderLoss]
  );

  const stop = useCallback(() => {
    runningRef.current = false;
    setIsRunning(false);
    if (animRef.current !== null) {
      cancelAnimationFrame(animRef.current);
      animRef.current = null;
    }
  }, []);

  const train = useCallback(() => {
    if (runningRef.current) {
      stop();
      return;
    }
    runningRef.current = true;
    setIsRunning(true);
    lastFrameTimeRef.current = 0;
    const interval = Math.max(16, 120 / speedScale);

    const loop = (ts: number) => {
      if (!runningRef.current) return;
      if (ts - lastFrameTimeRef.current >= interval) {
        lastFrameTimeRef.current = ts;
        doStep(lr);
        if (epochRef.current >= MAX_EPOCHS) {
          stop();
          return;
        }
      }
      animRef.current = requestAnimationFrame(loop);
    };
    animRef.current = requestAnimationFrame(loop);
  }, [speedScale, lr, doStep, stop]);

  const reset = useCallback(
    (type?: DatasetType, newLr?: number) => {
      stop();
      const t = type ?? datasetType;
      const eta = newLr ?? lr;
      ptsRef.current = generateDataset(t);
      modelRef.current = new LogisticRegression(2, { lr: eta, lambda: 0.001 });
      lossHistoryRef.current = [];
      epochRef.current = 0;
      setEpoch(0);
      setCurrentLoss(null);
      setAccuracy(null);
      // Defer render to after React re-paint
      requestAnimationFrame(() => {
        render();
        renderLoss();
      });
    },
    [stop, datasetType, lr, generateDataset, render, renderLoss]
  );

  // Initial mount
  useEffect(() => {
    reset("blobs", 0.3);
    return () => stop();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const handleDatasetChange = (type: DatasetType) => {
    setDatasetType(type);
    reset(type);
  };

  const CANVAS_W = 480;
  const CANVAS_H = 420;
  const LOSS_H = 72;

  return (
    <div
      style={{
        display: "flex",
        gap: 24,
        padding: "16px 20px",
        fontFamily: "system-ui, -apple-system, sans-serif",
        maxWidth: 920,
        margin: "0 auto",
        alignItems: "flex-start",
        color: T.text,
      }}
    >
      {/* ── Left column: canvas + loss curve ── */}
      <div
        style={{
          flex: "0 0 auto",
          display: "flex",
          flexDirection: "column",
          gap: 8,
        }}
      >
        <canvas
          ref={canvasRef}
          width={CANVAS_W}
          height={CANVAS_H}
          style={{
            border: T.border,
            borderRadius: 10,
            display: "block",
            boxShadow: `0 2px 12px ${T.shadow}`,
          }}
        />
        <div>
          <div
            style={{
              fontSize: 11,
              fontWeight: 600,
              color: T.subText,
              marginBottom: 3,
              letterSpacing: "0.04em",
              textTransform: "uppercase",
            }}
          >
            Training Loss
          </div>
          <canvas
            ref={lossCanvasRef}
            width={CANVAS_W}
            height={LOSS_H}
            style={{
              border: T.border,
              borderRadius: 6,
              display: "block",
            }}
          />
        </div>
      </div>

      {/* ── Right column: controls ── */}
      <div
        style={{
          flex: 1,
          display: "flex",
          flexDirection: "column",
          gap: 18,
          minWidth: 200,
        }}
      >
        {/* Title */}
        <div>
          <h2
            style={{
              margin: "0 0 6px",
              fontSize: 22,
              fontWeight: 700,
              color: T.titleText,
            }}
          >
            Logistic Regression
          </h2>
          <p
            style={{
              margin: 0,
              fontSize: 13,
              color: T.subText,
              lineHeight: 1.6,
            }}
          >
            Binary classifier using the sigmoid function. Minimises
            cross-entropy loss via gradient descent. The dashed line is the
            decision boundary (P = 0.5).
          </p>
        </div>

        {/* Dataset selector */}
        <div style={{ display: "flex", flexDirection: "column", gap: 7 }}>
          <span
            style={{ fontSize: 12, fontWeight: 600, color: T.labelText }}
          >
            Dataset
          </span>
          <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
            {(["blobs", "moons", "circles", "xor", "spirals"] as DatasetType[]).map(
              (t) => (
                <button
                  key={t}
                  onClick={() => handleDatasetChange(t)}
                  style={{
                    padding: "5px 12px",
                    fontSize: 12,
                    fontWeight: 600,
                    borderRadius: 6,
                    border: "1px solid",
                    cursor: "pointer",
                    background: datasetType === t ? T.accent : T.btnBg,
                    color: datasetType === t ? "#fff" : T.btnText,
                    borderColor: datasetType === t ? T.accent : (isDark ? "rgba(255,255,255,0.15)" : "#cbd5e0"),
                    transition: "all 0.15s",
                  }}
                >
                  {t.charAt(0).toUpperCase() + t.slice(1)}
                </button>
              )
            )}
          </div>
        </div>

        {/* Learning rate */}
        <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
          <label style={{ fontSize: 12, fontWeight: 600, color: T.labelText }}>
            Learning Rate:{" "}
            <span style={{ fontFamily: "monospace" }}>{lr.toFixed(2)}</span>
          </label>
          <input
            type="range"
            min="0.01"
            max="1.0"
            step="0.01"
            value={lr}
            onChange={(e) => {
              const v = Number(e.target.value);
              setLr(v);
              modelRef.current.lr = v;
            }}
            style={{ width: "100%", accentColor: T.accent }}
          />
        </div>

        {/* Action buttons */}
        <div style={{ display: "flex", gap: 8 }}>
          <button
            onClick={train}
            style={{
              padding: "9px 22px",
              fontSize: 13,
              fontWeight: 700,
              borderRadius: 8,
              border: "none",
              cursor: "pointer",
              background: isRunning ? "#e53e3e" : T.accent,
              color: "#fff",
              boxShadow: "0 2px 8px rgba(0,0,0,0.15)",
              transition: "background 0.15s",
            }}
          >
            {isRunning ? "⏸ Pause" : "▶ Train"}
          </button>
          <button
            onClick={() => doStep(lr)}
            style={{
              padding: "9px 18px",
              fontSize: 13,
              fontWeight: 600,
              borderRadius: 8,
              border: T.border,
              cursor: "pointer",
              background: T.btnBg,
              color: T.btnText,
            }}
          >
            Step
          </button>
          <button
            onClick={() => reset()}
            style={{
              padding: "9px 18px",
              fontSize: 13,
              fontWeight: 600,
              borderRadius: 8,
              border: T.border,
              cursor: "pointer",
              background: T.btnBg,
              color: T.btnText,
            }}
          >
            Reset
          </button>
        </div>

        {/* Stats */}
        <div
          style={{
            padding: "12px 14px",
            background: T.statsBg,
            borderRadius: 10,
            border: T.border,
            display: "flex",
            flexDirection: "column",
            gap: 8,
          }}
        >
          {[
            { label: "Epoch", value: epoch.toString() },
            {
              label: "Loss",
              value:
                currentLoss !== null ? currentLoss.toFixed(4) : "—",
            },
            {
              label: "Accuracy",
              value:
                accuracy !== null
                  ? `${(accuracy * 100).toFixed(1)}%`
                  : "—",
            },
          ].map(({ label, value }) => (
            <div
              key={label}
              style={{
                display: "flex",
                justifyContent: "space-between",
                fontSize: 13,
              }}
            >
              <span style={{ color: T.subText }}>{label}</span>
              <span style={{ fontWeight: 700, fontFamily: "monospace", color: T.text }}>
                {value}
              </span>
            </div>
          ))}
        </div>

        {/* Math explanation */}
        <div
          style={{
            padding: "12px 14px",
            background: T.infoBg,
            borderRadius: 10,
            border: T.infoBorder,
            fontSize: 12,
            color: T.infoText,
            lineHeight: 1.7,
          }}
        >
          <strong>How it works</strong>
          <br />
          P(y=1 | x) = σ(w·x + b)
          <br />
          L = −[y·log(p) + (1−y)·log(1−p)]
          <br />
          Update: w ← w − α·∇L − α·λ·w
        </div>
      </div>
    </div>
  );
};

export default LogisticRegressionDemo;
