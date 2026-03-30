/**
 * Random Forest interactive demo (Canvas 2D).
 * Shows the ensemble vote-fraction heatmap as trees are added.
 */
import React, { useRef, useCallback, useState } from "react";
import RandomForest from "../utils/randomForest";
import { makeBlobs, makeMoons, makeCircles, makeXOR, makeSpirals, toXy, DataPoint } from "../utils/datasets";

type DatasetType = "blobs" | "moons" | "circles" | "xor" | "spirals";
interface ThemeColors { text?: string; headerBg?: string; controlBg?: string; accent?: string; shadow?: string; textMuted?: string; }
interface Props { speedScale?: number; theme?: ThemeColors; }

const RANGE = 1.3;
const CANVAS_W = 480;
const CANVAS_H = 420;
const toPx = (v: number, s: number) => ((v + RANGE) / (2 * RANGE)) * s;
const fromPx = (p: number, s: number) => (p / s) * (2 * RANGE) - RANGE;

const COLOR_A = [6, 182, 212] as const;    // cyan-500
const COLOR_B = [244, 63, 94] as const;    // rose-500
const MID     = [100, 116, 139] as const;  // slate-500 midpoint

function blend(prob: number) {
  const t = Math.max(0, Math.min(1, prob));
  if (t >= 0.5) {
    const s = (t - 0.5) * 2;
    return [Math.round(COLOR_A[0] * s + MID[0] * (1 - s)), Math.round(COLOR_A[1] * s + MID[1] * (1 - s)), Math.round(COLOR_A[2] * s + MID[2] * (1 - s))];
  }
  const s = (0.5 - t) * 2;
  return [Math.round(COLOR_B[0] * s + MID[0] * (1 - s)), Math.round(COLOR_B[1] * s + MID[1] * (1 - s)), Math.round(COLOR_B[2] * s + MID[2] * (1 - s))];
}

function splitDataset(pts: DataPoint[]) {
  const sh = [...pts].sort(() => Math.random() - 0.5);
  const testN = Math.max(1, Math.round(sh.length * 0.2));
  return sh.map((p, i) => ({ ...p, isTest: i < testN }));
}

function computeMetrics(pts: DataPoint[], predict: (x: number, y: number) => number, testOnly: boolean) {
  const sub = testOnly ? pts.filter(p => p.isTest) : pts.filter(p => !p.isTest);
  if (!sub.length) return null;
  let tp = 0, tn = 0, fp = 0, fn = 0;
  for (const pt of sub) {
    const pr = predict(pt.x, pt.y);
    if (pr === 1 && pt.label === 1) tp++;
    else if (pr === 0 && pt.label === 0) tn++;
    else if (pr === 1 && pt.label === 0) fp++;
    else fn++;
  }
  const acc = (tp + tn) / sub.length;
  const prec = tp + fp > 0 ? tp / (tp + fp) : 0;
  const rec  = tp + fn > 0 ? tp / (tp + fn) : 0;
  const f1   = prec + rec > 0 ? 2 * prec * rec / (prec + rec) : 0;
  return { tp, tn, fp, fn, acc, prec, rec, f1 };
}

function genData(type: DatasetType, noise: number): DataPoint[] {
  switch (type) {
    case "moons":   return makeMoons(200, noise);
    case "circles": return makeCircles(200, noise, 0.4);
    case "xor":     return makeXOR(200, noise);
    case "spirals": return makeSpirals(200, noise * 0.5);
    default:        return makeBlobs(200, noise);
  }
}

const RandomForestDemo: React.FC<Props> = ({ theme }) => {
  const isDark = theme?.text === "#f7fafc";
  const T = {
    text: theme?.text ?? "#2d3748",
    sub: theme?.textMuted ?? "#718096",
    bg: theme?.controlBg ?? "rgba(255,255,255,0.98)",
    acc: theme?.accent ?? "#f43f5e",
    sh: theme?.shadow ?? "rgba(0,0,0,0.08)",
    brd: `1px solid ${isDark ? "rgba(255,255,255,0.12)" : "#e2e8f0"}`,
    statBg: isDark ? "rgba(45,55,72,0.6)" : "#f7fafc",
    insightBg: isDark ? "rgba(39,103,73,0.2)" : "#f0fff4",
    insightBorder: isDark ? "1px solid rgba(154,230,180,0.3)" : "1px solid #9ae6b4",
    insightText: isDark ? "#68d391" : "#276749",
    dimText: isDark ? "#718096" : "#a0aec0",
  };

  const canvasRef   = useRef<HTMLCanvasElement>(null);
  const rfRef       = useRef(new RandomForest({ nEstimators: 0, maxDepth: 5 }));
  const ptsRef      = useRef<DataPoint[]>([]);
  const animRef     = useRef<number | null>(null);
  const mousePosRef = useRef<{ px: number; py: number } | null>(null);

  const [dsType, setDsType]       = useState<DatasetType>("moons");
  const [noise, setNoise]         = useState(0.1);
  const [nTrees, setNTrees]       = useState(20);
  const [maxDepth, setMaxDepth]   = useState(5);
  const [trainTest, setTrainTest] = useState(true);
  const [treeCount, setTreeCount] = useState(0);
  const [growing, setGrowing]     = useState(false);
  const [accuracy, setAccuracy]   = useState<number | null>(null);
  const [testM, setTestM]       = useState<ReturnType<typeof computeMetrics>>(null);
  const [trainM, setTrainM]     = useState<ReturnType<typeof computeMetrics>>(null);
  const [dataChanged, setDataChanged] = useState(false);

  const drawCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const rf  = rfRef.current;
    const pts = ptsRef.current;
    const W = CANVAS_W, H = CANVAS_H;
    ctx.clearRect(0, 0, W, H);
    // Canvas background
    ctx.fillStyle = isDark ? "#0f172a" : "#f1f5f9";
    ctx.fillRect(0, 0, W, H);

    // Heatmap (before grid so grid is visible on top)
    const step = 4;
    const img = ctx.createImageData(W, H);
    const d = img.data;
    if (rf.trees.length > 0) {
      for (let px = 0; px < W; px += step) {
        for (let py = 0; py < H; py += step) {
          const prob = rf.predictProba([fromPx(px, W), fromPx(py, H)]);
          const [r, g, b] = blend(prob);
          for (let dx = 0; dx < step && px + dx < W; dx++) {
            for (let dy = 0; dy < step && py + dy < H; dy++) {
              const i = ((py + dy) * W + (px + dx)) * 4;
              d[i] = r; d[i+1] = g; d[i+2] = b; d[i+3] = 130;
            }
          }
        }
      }
    }
    ctx.putImageData(img, 0, 0);

    // Subtle grid (after heatmap so it's visible)
    ctx.strokeStyle = isDark ? "rgba(255,255,255,0.06)" : "rgba(0,0,0,0.06)";
    ctx.lineWidth = 1;
    const gridStep = W / 8;
    for (let gx = gridStep; gx < W; gx += gridStep) {
      ctx.beginPath(); ctx.moveTo(gx, 0); ctx.lineTo(gx, H); ctx.stroke();
    }
    for (let gy = gridStep; gy < H; gy += gridStep) {
      ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(W, gy); ctx.stroke();
    }

    // Points with glow
    for (const pt of pts) {
      const px = toPx(pt.x, W), py = toPx(pt.y, H);
      const [r, g, b] = pt.label === 1 ? COLOR_A : COLOR_B;
      const pointColor = `rgb(${r},${g},${b})`;

      ctx.shadowBlur = 14;
      ctx.shadowColor = pointColor;

      if (pt.isTest) {
        ctx.fillStyle = pointColor;
        ctx.strokeStyle = "rgba(255,255,255,0.9)";
        ctx.lineWidth = 1.5;
        ctx.fillRect(px - 5.5, py - 5.5, 11, 11);
        ctx.strokeRect(px - 5.5, py - 5.5, 11, 11);
      } else {
        ctx.beginPath();
        ctx.arc(px, py, 5.5, 0, Math.PI * 2);
        ctx.fillStyle = pointColor;
        ctx.fill();
        ctx.strokeStyle = "rgba(255,255,255,0.9)";
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }

      ctx.shadowBlur = 0;
    }
  }, [isDark]);

  const updateStats = useCallback(() => {
    const rf = rfRef.current;
    const pts = ptsRef.current;
    const { X, y } = toXy(pts.filter(p => !p.isTest));
    if (X.length) setAccuracy(rf.accuracy(X, y));
    const pfn = (x: number, yv: number) => rf.predictSample([x, yv]);
    setTrainM(computeMetrics(pts, pfn, false));
    setTestM(computeMetrics(pts, pfn, true));
  }, []);

  const stopGrowing = useCallback(() => {
    setGrowing(false);
    if (animRef.current !== null) { cancelAnimationFrame(animRef.current); animRef.current = null; }
  }, []);

  const growForest = useCallback(() => {
    const rf = rfRef.current;
    const pts = ptsRef.current;
    const { X, y } = toXy(pts.filter(p => !p.isTest));
    if (!X.length) return;

    rf.maxDepth = maxDepth;
    rf.reset();
    setTreeCount(0);
    setAccuracy(null);
    setTestM(null);
    setTrainM(null);
    setDataChanged(false);
    setGrowing(true);

    let added = 0;
    const addNext = () => {
      if (added >= nTrees) { stopGrowing(); updateStats(); return; }
      rf.addTree(X, y);
      added++;
      setTreeCount(added);
      drawCanvas();
      animRef.current = requestAnimationFrame(addNext);
    };
    animRef.current = requestAnimationFrame(addNext);
  }, [maxDepth, nTrees, drawCanvas, stopGrowing, updateStats]);

  const loadDataset = useCallback((type: DatasetType, n: number, split: boolean) => {
    stopGrowing();
    rfRef.current.reset();
    setTreeCount(0); setAccuracy(null); setTestM(null); setTrainM(null);
    let pts = genData(type, n);
    if (split) pts = splitDataset(pts);
    ptsRef.current = pts;
    drawCanvas();
  }, [stopGrowing, drawCanvas]);

  // Initial load — auto-start growing
  React.useEffect(() => {
    let pts = genData(dsType, noise);
    if (trainTest) pts = splitDataset(pts);
    ptsRef.current = pts;
    drawCanvas();
    setTimeout(() => growForest(), 80);
    return () => stopGrowing();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // R / B keyboard shortcuts — add point at current mouse position
  React.useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const key = e.key.toLowerCase();
      if (key !== "r" && key !== "b") return;
      e.preventDefault();
      const pos = mousePosRef.current;
      if (!pos) return;
      ptsRef.current = [...ptsRef.current, {
        x: fromPx(pos.px, CANVAS_W),
        y: fromPx(pos.py, CANVAS_H),
        label: key === "b" ? 1 : 0,
      }];
      setDataChanged(true);
      drawCanvas();
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [drawCanvas]);

  const fmtPct = (v: number | null | undefined) => v == null ? "—" : (v * 100).toFixed(1) + "%";

  const btnStyle = (active = false) => ({
    padding: "5px 10px", fontSize: 11, fontWeight: 600 as const,
    borderRadius: 6, cursor: "pointer" as const,
    background: active ? T.acc : T.bg, color: active ? "#fff" : T.text,
    border: T.brd, boxShadow: `0 1px 3px ${T.sh}`,
    transition: "all 0.15s",
  });
  const DATASETS: DatasetType[] = ["blobs", "moons", "circles", "xor", "spirals"];

  return (
    <div style={{ display: "flex", gap: 24, alignItems: "flex-start", flexWrap: "wrap", justifyContent: "center",
      fontFamily: "system-ui, -apple-system, sans-serif", color: T.text }}>

      {/* Canvas */}
      <div style={{ flex: "0 0 auto" }}>
        <canvas ref={canvasRef} width={CANVAS_W} height={CANVAS_H}
          style={{ border: T.brd, borderRadius: 10, display: "block", boxShadow: `0 2px 12px ${T.sh}`, cursor: "crosshair" }}
          onMouseMove={(e) => {
            const rect = (e.target as HTMLCanvasElement).getBoundingClientRect();
            mousePosRef.current = {
              px: (e.clientX - rect.left) * (CANVAS_W / rect.width),
              py: (e.clientY - rect.top)  * (CANVAS_H / rect.height),
            };
          }}
          onClick={(e) => {
            const rect = (e.target as HTMLCanvasElement).getBoundingClientRect();
            const px = (e.clientX - rect.left) * (CANVAS_W / rect.width);
            const py = (e.clientY - rect.top)  * (CANVAS_H / rect.height);
            ptsRef.current = [...ptsRef.current, { x: fromPx(px, CANVAS_W), y: fromPx(py, CANVAS_H), label: 1 }];
            setDataChanged(true);
            drawCanvas();
          }}
          onContextMenu={(e) => {
            e.preventDefault();
            const rect = (e.target as HTMLCanvasElement).getBoundingClientRect();
            const px = (e.clientX - rect.left) * (CANVAS_W / rect.width);
            const py = (e.clientY - rect.top)  * (CANVAS_H / rect.height);
            ptsRef.current = [...ptsRef.current, { x: fromPx(px, CANVAS_W), y: fromPx(py, CANVAS_H), label: 0 }];
            setDataChanged(true);
            drawCanvas();
          }}
        />
        {dataChanged && (
          <div style={{ marginTop: 4, padding: "5px 10px", borderRadius: 6, fontSize: 11, fontWeight: 600,
            background: isDark ? "rgba(245,158,11,0.15)" : "#fffbeb",
            border: `1px solid ${isDark ? "rgba(245,158,11,0.4)" : "#fcd34d"}`,
            color: isDark ? "#fcd34d" : "#92400e" }}>
            Dataset changed — click Grow Forest to rebuild
          </div>
        )}
        <div style={{ marginTop: 6, display: "flex", gap: 14, fontSize: 11, color: T.sub, flexWrap: "wrap" as const }}>
          <span style={{ color: "#06b6d4" }}>● blue — left-click or B</span>
          <span style={{ color: "#f43f5e" }}>● red — right-click or R</span>
          <span style={{ marginLeft: "auto" }}>Trees: {treeCount}/{growing ? "…" : nTrees}</span>
        </div>
      </div>

      {/* Controls */}
      <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: 16, minWidth: 210 }}>
        {/* Title */}
        <div>
          <h2 style={{ margin: "0 0 6px", fontSize: 22, fontWeight: 700, color: T.text }}>
            Random Forest
          </h2>
          <p style={{ margin: 0, fontSize: 13, color: T.sub, lineHeight: 1.6 }}>
            An ensemble of CART trees, each trained on a bootstrap sample.
            Classification by majority vote — watch the boundary stabilise as trees are added.
          </p>
        </div>

        {/* Dataset */}
        <div>
          <div style={{ fontSize: 12, fontWeight: 600, color: T.text, marginBottom: 6 }}>Dataset</div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 5 }}>
            {DATASETS.map(ds => (
              <button key={ds} style={btnStyle(dsType === ds)}
                onClick={() => { setDsType(ds); loadDataset(ds, noise, trainTest); }}>
                {ds.charAt(0).toUpperCase() + ds.slice(1)}
              </button>
            ))}
          </div>
        </div>

        {/* Noise */}
        <div>
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, color: T.text, fontWeight: 600, marginBottom: 4 }}>
            <span>Noise</span><span style={{ color: T.sub, fontFamily: "monospace" }}>{noise.toFixed(2)}</span>
          </div>
          <input type="range" min="0" max="0.3" step="0.01" value={noise} style={{ width: "100%", accentColor: T.acc }}
            onChange={e => { const v = Number(e.target.value); setNoise(v); loadDataset(dsType, v, trainTest); }} />
        </div>

        {/* Trees */}
        <div>
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, color: T.text, fontWeight: 600, marginBottom: 4 }}>
            <span>Trees</span><span style={{ color: T.sub, fontFamily: "monospace" }}>{nTrees}</span>
          </div>
          <input type="range" min="1" max="60" step="1" value={nTrees} style={{ width: "100%", accentColor: T.acc }}
            onChange={e => setNTrees(Number(e.target.value))} />
        </div>

        {/* Max Depth */}
        <div>
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, color: T.text, fontWeight: 600, marginBottom: 4 }}>
            <span>Max Depth</span><span style={{ color: T.sub, fontFamily: "monospace" }}>{maxDepth}</span>
          </div>
          <input type="range" min="1" max="10" step="1" value={maxDepth} style={{ width: "100%", accentColor: T.acc }}
            onChange={e => setMaxDepth(Number(e.target.value))} />
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, color: T.dimText, marginTop: 2 }}>
            <span>shallow</span><span>deep</span>
          </div>
        </div>

        {/* Train/test */}
        <label style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 12, color: T.text, cursor: "pointer" }}>
          <input type="checkbox" checked={trainTest} style={{ accentColor: T.acc }}
            onChange={e => { setTrainTest(e.target.checked); loadDataset(dsType, noise, e.target.checked); }} />
          Train/test split (20%)
        </label>

        {/* Buttons */}
        <div style={{ display: "flex", gap: 8 }}>
          <button style={{ padding: "9px 18px", fontSize: 13, fontWeight: 700, borderRadius: 8, border: "none",
            cursor: "pointer", background: growing ? "#e53e3e" : T.acc, color: "#fff",
            boxShadow: "0 2px 8px rgba(0,0,0,0.12)", flex: 1 }}
            onClick={() => growing ? stopGrowing() : growForest()}>
            {growing ? "⏸ Stop" : "🌲 Grow Forest"}
          </button>
          <button style={{ ...btnStyle(), padding: "9px 18px", fontSize: 13 }}
            onClick={() => loadDataset(dsType, noise, trainTest)}>
            Reset
          </button>
        </div>

        {/* Stats */}
        <div style={{ padding: "12px 14px", background: T.statBg, borderRadius: 10, border: T.brd }}>
          <div style={{ fontSize: 11, fontWeight: 700, color: T.sub, marginBottom: 8, textTransform: "uppercase" as const, letterSpacing: "0.05em" }}>
            Forest Stats
          </div>
          {[
            { label: "Trees grown", value: treeCount || "—" },
            { label: "Train accuracy", value: fmtPct(accuracy) },
            { label: "Test accuracy", value: fmtPct(testM?.acc) },
          ].map(({ label, value }) => (
            <div key={label} style={{ display: "flex", justifyContent: "space-between", fontSize: 13, marginBottom: 4 }}>
              <span style={{ color: T.sub }}>{label}</span>
              <span style={{ fontWeight: 700, fontFamily: "monospace", color: T.text }}>{String(value)}</span>
            </div>
          ))}
        </div>

        {/* Confusion matrix */}
        {(testM ?? trainM) && (() => {
          const m = testM ?? trainM;
          if (!m) return null;
          return (
            <div style={{ padding: "12px 14px", background: T.statBg, borderRadius: 10, border: T.brd }}>
              <div style={{ fontSize: 11, fontWeight: 700, color: T.sub, marginBottom: 8, textTransform: "uppercase" as const, letterSpacing: "0.05em" }}>
                {testM ? "Test Metrics" : "Train Metrics"}
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "auto 1fr 1fr", gap: 2, fontSize: 11, marginBottom: 8 }}>
                <span/><span style={{ textAlign: "center" as const, color: T.sub, fontWeight: 600 }}>Pred 0</span><span style={{ textAlign: "center" as const, color: T.sub, fontWeight: 600 }}>Pred 1</span>
                <span style={{ color: T.sub, fontWeight: 600 }}>Act 0</span>
                <span style={{ textAlign: "center" as const, background: isDark ? "rgba(72,187,120,0.2)" : "#f0fff4", borderRadius: 4, padding: "2px 4px", fontWeight: 700, color: isDark ? "#68d391" : "#276749" }}>{m.tn}</span>
                <span style={{ textAlign: "center" as const, background: isDark ? "rgba(252,129,129,0.2)" : "#fff5f5", borderRadius: 4, padding: "2px 4px", fontWeight: 700, color: isDark ? "#fc8181" : "#c53030" }}>{m.fp}</span>
                <span style={{ color: T.sub, fontWeight: 600 }}>Act 1</span>
                <span style={{ textAlign: "center" as const, background: isDark ? "rgba(252,129,129,0.2)" : "#fff5f5", borderRadius: 4, padding: "2px 4px", fontWeight: 700, color: isDark ? "#fc8181" : "#c53030" }}>{m.fn}</span>
                <span style={{ textAlign: "center" as const, background: isDark ? "rgba(72,187,120,0.2)" : "#f0fff4", borderRadius: 4, padding: "2px 4px", fontWeight: 700, color: isDark ? "#68d391" : "#276749" }}>{m.tp}</span>
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "3px 8px", fontSize: 12, color: T.text }}>
                <span style={{ color: T.sub }}>Precision</span><span style={{ fontWeight: 700, fontFamily: "monospace" }}>{fmtPct(m.prec)}</span>
                <span style={{ color: T.sub }}>Recall</span><span style={{ fontWeight: 700, fontFamily: "monospace" }}>{fmtPct(m.rec)}</span>
                <span style={{ color: T.sub }}>F1</span><span style={{ fontWeight: 700, fontFamily: "monospace" }}>{fmtPct(m.f1)}</span>
              </div>
            </div>
          );
        })()}

        {/* How it works */}
        <div style={{ padding: "12px 14px", background: T.insightBg, borderRadius: 10,
          border: T.insightBorder, fontSize: 12, color: T.insightText, lineHeight: 1.7 }}>
          <strong>How it works</strong><br />
          Each tree trains on a bootstrap sample (random sample with replacement).<br />
          The boundary = majority vote of all trees.<br />
          More trees → lower variance, smoother boundary.
        </div>

        <div style={{ fontSize: 11, color: T.dimText, lineHeight: 1.6 }}>
          <strong>Tip:</strong> try increasing depth on spirals — a single deep tree overfits,
          but the forest smooths it out.
        </div>
      </div>
    </div>
  );
};

export default RandomForestDemo;
