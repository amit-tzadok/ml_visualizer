/**
 * Decision Tree interactive demo.
 * Uses the HTML Canvas 2D API (no p5.js) for rendering.
 * Visualises axis-aligned splits growing depth by depth with animated region colouring.
 */
import React, {
  useRef,
  useEffect,
  useCallback,
  useState,
} from "react";
import DecisionTree, { SplitBoundary } from "../utils/decisionTree";
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

interface DecisionTreeDemoProps {
  speedScale?: number;
  showInstructions?: boolean;
  theme?: ThemeColors;
}

// Coordinate helpers: maps [-1.3, 1.3] <-> [0, canvasSize]
const RANGE = 1.3;
const toPx = (v: number, size: number) => ((v + RANGE) / (2 * RANGE)) * size;
const fromPx = (px: number, size: number) => (px / size) * (2 * RANGE) - RANGE;

// Depth-indexed split colours
const SPLIT_COLORS = [
  "#4a5568",
  "#2b6cb0",
  "#276749",
  "#744210",
  "#553c9a",
  "#97266d",
];

const CLASS_COLORS: Record<number, string> = {
  0: "rgba(252,129,129,0.35)",
  1: "rgba(66,153,225,0.35)",
};

const PT_COLORS: Record<number, string> = {
  0: "rgb(229,62,62)",
  1: "rgb(49,130,206)",
};

const CANVAS_W = 480;
const CANVAS_H = 420;

function generateDataset(type: DatasetType): DataPoint[] {
  switch (type) {
    case "moons":   return makeMoons(180, 0.1);
    case "circles": return makeCircles(180, 0.06, 0.4);
    case "xor":     return makeXOR(200, 0.07);
    case "spirals": return makeSpirals(200, 0.04);
    default:        return makeBlobs(180, 0.12);
  }
}

const DecisionTreeDemo: React.FC<DecisionTreeDemoProps> = ({
  speedScale = 1,
  theme,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animRef = useRef<number | null>(null);
  const treeRef = useRef(new DecisionTree({ maxDepth: 1 }));
  const ptsRef = useRef<DataPoint[]>([]);
  const boundariesRef = useRef<SplitBoundary[]>([]);
  const visibleBoundariesRef = useRef<number>(0); // how many splits are drawn
  const animatingRef = useRef(false);

  const [datasetType, setDatasetType] = useState<DatasetType>("blobs");
  const [maxDepth, setMaxDepth] = useState(4);
  const [accuracy, setAccuracy] = useState<number | null>(null);
  const [treeDepth, setTreeDepth] = useState(0);
  const [splitCount, setSplitCount] = useState(0);
  const [isAnimating, setIsAnimating] = useState(false);

  // Derived theme tokens — fall back to light-mode literals when no theme is provided
  const isDark = theme?.text === "#f7fafc";
  const T = {
    text:          theme?.text      ?? "#2d3748",
    subText:       theme?.textMuted ?? "#718096",
    labelText:     isDark ? "#cbd5e0" : "#4a5568",
    titleText:     theme?.text      ?? "#1a202c",
    controlBg:     theme?.controlBg ?? "rgba(255,255,255,0.98)",
    accent:        theme?.accent    ?? "#f43f5e",
    shadow:        theme?.shadow    ?? "rgba(0,0,0,0.08)",
    border:        `1px solid ${isDark ? "rgba(255,255,255,0.12)" : "#e2e8f0"}`,
    statsBg:       isDark ? "rgba(45,55,72,0.6)"      : "#f7fafc",
    insightBg:     isDark ? "rgba(39,103,73,0.2)"     : "#f0fff4",
    insightBorder: isDark ? "1px solid rgba(154,230,180,0.3)" : "1px solid #9ae6b4",
    insightText:   isDark ? "#68d391"                 : "#276749",
    dimText:       isDark ? "#718096"                 : "#a0aec0",
    btnBg:         theme?.controlBg ?? "#fff",
    btnText:       theme?.text      ?? "#4a5568",
  };

  const drawRegions = useCallback(
    (ctx: CanvasRenderingContext2D, tree: DecisionTree) => {
      const W = CANVAS_W;
      const H = CANVAS_H;
      if (!tree.root) return;
      const step = 4;
      const imageData = ctx.createImageData(W, H);
      const data = imageData.data;
      for (let px = 0; px < W; px += step) {
        for (let py = 0; py < H; py += step) {
          const x = fromPx(px, W);
          const y = fromPx(py, H);
          const pred = tree.predictSample([x, y]);
          const color = pred === 1 ? [66, 153, 225] : [252, 129, 129];
          for (let dx = 0; dx < step && px + dx < W; dx++) {
            for (let dy = 0; dy < step && py + dy < H; dy++) {
              const idx = ((py + dy) * W + (px + dx)) * 4;
              data[idx]     = color[0];
              data[idx + 1] = color[1];
              data[idx + 2] = color[2];
              data[idx + 3] = 50;
            }
          }
        }
      }
      ctx.putImageData(imageData, 0, 0);
    },
    []
  );

  const drawSplits = useCallback(
    (ctx: CanvasRenderingContext2D, boundaries: SplitBoundary[], count: number) => {
      for (let i = 0; i < count && i < boundaries.length; i++) {
        const b = boundaries[i];
        const color = SPLIT_COLORS[Math.min(b.depth, SPLIT_COLORS.length - 1)];
        ctx.strokeStyle = color;
        ctx.lineWidth = Math.max(1, 2.5 - b.depth * 0.4);
        ctx.beginPath();

        if (b.feature === 0) {
          // Vertical split
          const px = toPx(b.threshold, CANVAS_W);
          const py0 = toPx(b.yMin, CANVAS_H);
          const py1 = toPx(b.yMax, CANVAS_H);
          ctx.moveTo(px, py0);
          ctx.lineTo(px, py1);
        } else {
          // Horizontal split
          const py = toPx(b.threshold, CANVAS_H);
          const px0 = toPx(b.xMin, CANVAS_W);
          const px1 = toPx(b.xMax, CANVAS_W);
          ctx.moveTo(px0, py);
          ctx.lineTo(px1, py);
        }
        ctx.stroke();
      }
    },
    []
  );

  const drawPoints = useCallback((ctx: CanvasRenderingContext2D, pts: DataPoint[]) => {
    for (const pt of pts) {
      const px = toPx(pt.x, CANVAS_W);
      const py = toPx(pt.y, CANVAS_H);
      ctx.beginPath();
      ctx.arc(px, py, 4.5, 0, Math.PI * 2);
      ctx.fillStyle = PT_COLORS[pt.label] ?? "#999";
      ctx.fill();
      ctx.strokeStyle = "rgba(255,255,255,0.7)";
      ctx.lineWidth = 1.2;
      ctx.stroke();
    }
  }, []);

  const render = useCallback(
    (visibleCount?: number) => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      const count =
        visibleCount !== undefined
          ? visibleCount
          : visibleBoundariesRef.current;
      ctx.clearRect(0, 0, CANVAS_W, CANVAS_H);
      drawRegions(ctx, treeRef.current);
      drawSplits(ctx, boundariesRef.current, count);
      drawPoints(ctx, ptsRef.current);
    },
    [drawRegions, drawSplits, drawPoints]
  );

  const stopAnimation = useCallback(() => {
    animatingRef.current = false;
    setIsAnimating(false);
    if (animRef.current !== null) {
      cancelAnimationFrame(animRef.current);
      animRef.current = null;
    }
  }, []);

  const buildAndAnimate = useCallback(
    (depth: number, pts: DataPoint[]) => {
      stopAnimation();
      const { X, y } = toXy(pts);
      const tree = new DecisionTree({
        maxDepth: depth,
        minSamplesSplit: 4,
        minSamplesLeaf: 2,
      });
      tree.fit(X, y);
      treeRef.current = tree;
      const bounds = tree.getSplitBoundaries();
      boundariesRef.current = bounds;
      const acc = tree.accuracy(X, y);
      const td = tree.treeDepth();
      setAccuracy(acc);
      setTreeDepth(td);
      setSplitCount(bounds.length);

      if (bounds.length === 0) {
        visibleBoundariesRef.current = 0;
        render(0);
        return;
      }

      // Animate splits appearing one by one
      animatingRef.current = true;
      setIsAnimating(true);
      visibleBoundariesRef.current = 0;

      const delay = Math.max(40, 600 / (speedScale * Math.max(bounds.length, 1)));
      let lastTime = 0;

      const loop = (ts: number) => {
        if (!animatingRef.current) return;
        if (ts - lastTime >= delay) {
          lastTime = ts;
          visibleBoundariesRef.current++;
          render(visibleBoundariesRef.current);
          if (visibleBoundariesRef.current >= bounds.length) {
            stopAnimation();
            return;
          }
        }
        animRef.current = requestAnimationFrame(loop);
      };
      animRef.current = requestAnimationFrame(loop);
    },
    [render, stopAnimation, speedScale]
  );

  const reset = useCallback(
    (type?: DatasetType, depth?: number) => {
      stopAnimation();
      const t = type ?? datasetType;
      const d = depth ?? maxDepth;
      ptsRef.current = generateDataset(t);
      treeRef.current = new DecisionTree({ maxDepth: d });
      boundariesRef.current = [];
      visibleBoundariesRef.current = 0;
      setAccuracy(null);
      setTreeDepth(0);
      setSplitCount(0);
      requestAnimationFrame(() => render(0));
    },
    [stopAnimation, datasetType, maxDepth, render]
  );

  // Initial mount
  useEffect(() => {
    reset("blobs", 4);
    return () => stopAnimation();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const handleDatasetChange = (type: DatasetType) => {
    setDatasetType(type);
    reset(type);
  };

  const handleDepthChange = (d: number) => {
    setMaxDepth(d);
  };

  const handleBuild = () => {
    buildAndAnimate(maxDepth, ptsRef.current);
  };

  // Suppress unused variable warnings for CLASS_COLORS (used implicitly by canvas)
  void CLASS_COLORS;

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
      {/* ── Canvas ── */}
      <div style={{ flex: "0 0 auto" }}>
        <canvas
          ref={canvasRef}
          width={CANVAS_W}
          height={CANVAS_H}
          style={{
            border: T.border,
            borderRadius: 10,
            display: "block",
            boxShadow: `0 2px 12px ${T.shadow}`,
            cursor: "crosshair",
          }}
          onClick={(e) => {
            const rect = (e.target as HTMLCanvasElement).getBoundingClientRect();
            const px = (e.clientX - rect.left) * (CANVAS_W / rect.width);
            const py = (e.clientY - rect.top)  * (CANVAS_H / rect.height);
            ptsRef.current = [...ptsRef.current, { x: fromPx(px, CANVAS_W), y: fromPx(py, CANVAS_H), label: 1 }];
            buildAndAnimate(maxDepth, ptsRef.current);
          }}
          onContextMenu={(e) => {
            e.preventDefault();
            const rect = (e.target as HTMLCanvasElement).getBoundingClientRect();
            const px = (e.clientX - rect.left) * (CANVAS_W / rect.width);
            const py = (e.clientY - rect.top)  * (CANVAS_H / rect.height);
            ptsRef.current = [...ptsRef.current, { x: fromPx(px, CANVAS_W), y: fromPx(py, CANVAS_H), label: 0 }];
            buildAndAnimate(maxDepth, ptsRef.current);
          }}
        />
        {/* Click legend */}
        <div style={{ marginTop: 6, display: "flex", gap: 14, fontSize: 11, color: T.subText }}>
          <span style={{ color: "#4299e1" }}>● left-click: add blue</span>
          <span style={{ color: "#e53e3e" }}>● right-click: add red</span>
        </div>
        {/* Depth colour legend */}
        <div
          style={{
            marginTop: 6,
            display: "flex",
            gap: 10,
            flexWrap: "wrap",
            fontSize: 11,
          }}
        >
          {SPLIT_COLORS.slice(0, Math.min(maxDepth, 6)).map((c, i) => (
            <span key={i} style={{ display: "flex", alignItems: "center", gap: 4 }}>
              <span
                style={{
                  display: "inline-block",
                  width: 16,
                  height: 3,
                  background: c,
                  borderRadius: 2,
                }}
              />
              <span style={{ color: T.subText }}>depth {i}</span>
            </span>
          ))}
        </div>
      </div>

      {/* ── Controls ── */}
      <div
        style={{
          flex: 1,
          display: "flex",
          flexDirection: "column",
          gap: 18,
          minWidth: 200,
        }}
      >
        <div>
          <h2
            style={{
              margin: "0 0 6px",
              fontSize: 22,
              fontWeight: 700,
              color: T.titleText,
            }}
          >
            Decision Tree
          </h2>
          <p style={{ margin: 0, fontSize: 13, color: T.subText, lineHeight: 1.6 }}>
            CART-style tree using Gini impurity. Each split partitions the
            feature space with an axis-aligned line. Watch the splits grow depth
            by depth.
          </p>
        </div>

        {/* Dataset */}
        <div style={{ display: "flex", flexDirection: "column", gap: 7 }}>
          <span style={{ fontSize: 12, fontWeight: 600, color: T.labelText }}>
            Dataset
          </span>
          <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
            {(
              ["blobs", "moons", "circles", "xor", "spirals"] as DatasetType[]
            ).map((t) => (
              <button
                key={t}
                onClick={() => handleDatasetChange(t)}
                style={{
                  padding: "5px 10px",
                  fontSize: 11,
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
            ))}
          </div>
        </div>

        {/* Max depth */}
        <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
          <label style={{ fontSize: 12, fontWeight: 600, color: T.labelText }}>
            Max Depth:{" "}
            <span style={{ fontFamily: "monospace" }}>{maxDepth}</span>
          </label>
          <input
            type="range"
            min="1"
            max="8"
            step="1"
            value={maxDepth}
            onChange={(e) => handleDepthChange(Number(e.target.value))}
            style={{ width: "100%", accentColor: T.accent }}
          />
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              fontSize: 11,
              color: T.dimText,
            }}
          >
            <span>shallow</span>
            <span>deep</span>
          </div>
        </div>

        {/* Action buttons */}
        <div style={{ display: "flex", gap: 8 }}>
          <button
            onClick={handleBuild}
            disabled={isAnimating}
            style={{
              padding: "9px 22px",
              fontSize: 13,
              fontWeight: 700,
              borderRadius: 8,
              border: "none",
              cursor: isAnimating ? "not-allowed" : "pointer",
              background: isAnimating ? (isDark ? "#4a5568" : "#a0aec0") : T.accent,
              color: "#fff",
              boxShadow: "0 2px 8px rgba(0,0,0,0.12)",
              transition: "background 0.15s",
              opacity: isAnimating ? 0.7 : 1,
            }}
          >
            {isAnimating ? "Building…" : "🌲 Build Tree"}
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
            { label: "Tree Depth", value: treeDepth || "—" },
            { label: "Total Splits", value: splitCount || "—" },
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
                {String(value)}
              </span>
            </div>
          ))}
        </div>

        {/* Insight box */}
        <div
          style={{
            padding: "12px 14px",
            background: T.insightBg,
            borderRadius: 10,
            border: T.insightBorder,
            fontSize: 12,
            color: T.insightText,
            lineHeight: 1.7,
          }}
        >
          <strong>How it works</strong>
          <br />
          At each node, find the (feature, threshold) that maximises Gini gain.
          <br />
          Gini(S) = 1 − Σ pᵢ²&nbsp;&nbsp;|&nbsp;&nbsp;IG = Gini(S) − wL·Gini(L) − wR·Gini(R)
        </div>

        {/* Depth tip */}
        <div
          style={{
            fontSize: 11,
            color: T.dimText,
            lineHeight: 1.6,
          }}
        >
          <strong>Tip:</strong> shallow trees underfit; deep trees overfit.
          Watch how accuracy on training data increases with depth while
          the regions become more irregular.
        </div>
      </div>
    </div>
  );
};

export default DecisionTreeDemo;
