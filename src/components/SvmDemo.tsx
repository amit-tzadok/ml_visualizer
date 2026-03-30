/**
 * SVM interactive demo (Canvas 2D).
 * Supports Linear and RBF kernels. Shows decision boundary, margin band,
 * support vectors (linear), and a live hinge-loss curve.
 */
import React, { useRef, useEffect, useCallback, useState } from "react";
import SVM from "../utils/svm";
import {
  makeBlobs, makeMoons, makeCircles, makeXOR, makeSpirals, toXy, DataPoint,
} from "../utils/datasets";

type DatasetType = "blobs" | "moons" | "circles" | "xor" | "spirals";
type KernelType  = "linear" | "rbf";

interface ThemeColors {
  text?: string; headerBg?: string; controlBg?: string;
  accent?: string; shadow?: string; textMuted?: string;
}
interface SvmDemoProps { speedScale?: number; theme?: ThemeColors; }

const RANGE    = 1.3;
const CANVAS_W = 480;
const CANVAS_H = 420;
const LOSS_H   = 72;
const toPx   = (v: number, size: number) => ((v + RANGE) / (2 * RANGE)) * size;
const fromPx = (px: number, size: number) => (px / size) * (2 * RANGE) - RANGE;

const COLOR_A = [6, 182, 212] as const;    // cyan-500  — label 1
const COLOR_B = [244, 63, 94] as const;    // rose-500  — label 0
const MID     = [100, 116, 139] as const;  // slate-500 — uncertain midpoint

function blend(prob: number) {
  const t = Math.max(0, Math.min(1, prob));
  if (t >= 0.5) {
    const s = (t - 0.5) * 2;
    return [Math.round(COLOR_A[0]*s + MID[0]*(1-s)), Math.round(COLOR_A[1]*s + MID[1]*(1-s)), Math.round(COLOR_A[2]*s + MID[2]*(1-s))];
  }
  const s = (0.5 - t) * 2;
  return [Math.round(COLOR_B[0]*s + MID[0]*(1-s)), Math.round(COLOR_B[1]*s + MID[1]*(1-s)), Math.round(COLOR_B[2]*s + MID[2]*(1-s))];
}

function splitDataset(pts: DataPoint[], testFrac = 0.2): DataPoint[] {
  const sh   = [...pts].sort(() => Math.random() - 0.5);
  const testN = Math.max(1, Math.round(sh.length * testFrac));
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
  const acc  = (tp + tn) / sub.length;
  const prec = tp + fp > 0 ? tp / (tp + fp) : 0;
  const rec  = tp + fn > 0 ? tp / (tp + fn) : 0;
  const f1   = prec + rec > 0 ? 2 * prec * rec / (prec + rec) : 0;
  return { tp, tn, fp, fn, acc, prec, rec, f1 };
}

function generateDataset(type: DatasetType, noise: number): DataPoint[] {
  switch (type) {
    case "moons":   return makeMoons(180, noise);
    case "circles": return makeCircles(180, noise, 0.4);
    case "xor":     return makeXOR(200, noise);
    case "spirals": return makeSpirals(200, noise * 0.5);
    default:        return makeBlobs(180, noise);
  }
}

const SvmDemo: React.FC<SvmDemoProps> = ({ speedScale = 1, theme }) => {
  const isDark = theme?.text === "#f7fafc";
  const T = {
    text:    theme?.text      ?? "#2d3748",
    sub:     theme?.textMuted ?? "#718096",
    bg:      theme?.controlBg ?? "rgba(255,255,255,0.98)",
    acc:     theme?.accent    ?? "#f43f5e",
    sh:      theme?.shadow    ?? "rgba(0,0,0,0.08)",
    brd:     `1px solid ${isDark ? "rgba(255,255,255,0.12)" : "#e2e8f0"}`,
    statBg:  isDark ? "rgba(45,55,72,0.6)"          : "#f7fafc",
    insightBg:     isDark ? "rgba(39,103,73,0.2)"           : "#f0fff4",
    insightBorder: isDark ? "1px solid rgba(154,230,180,0.3)" : "1px solid #9ae6b4",
    insightText:   isDark ? "#68d391"                        : "#276749",
    dimText:       isDark ? "#718096"                        : "#a0aec0",
    lossBg:  isDark ? "rgba(45,55,72,0.95)" : "rgba(247,250,252,0.95)",
  };
  const themeRef = useRef(T);
  themeRef.current = T;

  const canvasRef    = useRef<HTMLCanvasElement>(null);
  const lossCanvasRef = useRef<HTMLCanvasElement>(null);
  const animRef      = useRef<number | null>(null);
  const svmRef       = useRef(new SVM({ C: 1, lr: 0.01, kernel: "linear" }));
  const ptsRef       = useRef<DataPoint[]>([]);
  const runningRef   = useRef(false);
  const epochRef     = useRef(0);
  const lastFrameRef = useRef(0);
  const lossHistRef  = useRef<number[]>([]);
  const mousePosRef  = useRef<{ px: number; py: number } | null>(null);

  const [dataset,        setDataset]        = useState<DatasetType>("blobs");
  const [noise,          setNoise]          = useState(0.12);
  const [C,              setC]              = useState(1.0);
  const [kernel,         setKernel]         = useState<KernelType>("linear");
  const [gamma,          setGamma]          = useState(1.0);
  const [isRunning,      setIsRunning]      = useState(false);
  const [epoch,          setEpoch]          = useState(0);
  const [accuracy,       setAccuracy]       = useState<number | null>(null);
  const [trainTestSplit, setTrainTestSplit] = useState(true);
  const [testMetrics,    setTestMetrics]    = useState<ReturnType<typeof computeMetrics>>(null);
  const [trainMetrics,   setTrainMetrics]   = useState<ReturnType<typeof computeMetrics>>(null);
  const [marginStr,      setMarginStr]      = useState<string>("-");

  // ── Loss curve renderer ───────────────────────────────────────────────────
  const renderLoss = useCallback(() => {
    const canvas = lossCanvasRef.current;
    if (!canvas) return;
    const ctx    = canvas.getContext("2d");
    if (!ctx)    return;
    const tc     = themeRef.current;
    const W      = canvas.width;
    const H      = canvas.height;
    const losses = lossHistRef.current;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = tc.lossBg;
    ctx.fillRect(0, 0, W, H);

    if (losses.length < 2) {
      ctx.fillStyle = tc.dimText;
      ctx.font = "12px system-ui";
      ctx.fillText("Start training to see hinge-loss curve", 16, H / 2 + 4);
      return;
    }
    const maxL  = losses[0];
    const minL  = Math.min(...losses);
    const range = Math.max(maxL - minL, 0.001);
    const pad   = { l: 10, r: 10, t: 8, b: 10 };
    const midY  = H - pad.b - ((maxL / 2 - minL) / range) * (H - pad.t - pad.b);
    ctx.strokeStyle = "rgba(203,213,224,0.5)";
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(pad.l, midY); ctx.lineTo(W - pad.r, midY); ctx.stroke();
    ctx.setLineDash([]);
    ctx.beginPath();
    for (let i = 0; i < losses.length; i++) {
      const px = pad.l + (i / Math.max(losses.length - 1, 1)) * (W - pad.l - pad.r);
      const py = H - pad.b - ((losses[i] - minL) / range) * (H - pad.t - pad.b);
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }
    ctx.strokeStyle = tc.acc;
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.fillStyle = tc.sub;
    ctx.font = "bold 11px system-ui";
    ctx.fillText(`Hinge loss: ${losses[losses.length - 1].toFixed(4)}`, pad.l + 2, 14);
  }, []);

  // ── Main canvas renderer ──────────────────────────────────────────────────
  const drawCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx)  return;
    const svm  = svmRef.current;
    const pts  = ptsRef.current;
    const W = CANVAS_W, H = CANVAS_H;
    ctx.clearRect(0, 0, W, H);
    // Canvas background
    ctx.fillStyle = isDark ? "#0f172a" : "#f1f5f9";
    ctx.fillRect(0, 0, W, H);

    // Heatmap (drawn before grid so grid renders on top)
    const step    = 4;
    const img     = ctx.createImageData(W, H);
    const d       = img.data;
    const trained = svm.weights.some(w => w !== 0);
    if (trained) {
      for (let px = 0; px < W; px += step) {
        for (let py = 0; py < H; py += step) {
          const prob = svm.predictProba([fromPx(px, W), fromPx(py, H)]);
          const [r, g, b] = blend(prob);
          for (let dx = 0; dx < step && px+dx < W; dx++) {
            for (let dy = 0; dy < step && py+dy < H; dy++) {
              const i = ((py+dy)*W + (px+dx)) * 4;
              d[i]=r; d[i+1]=g; d[i+2]=b; d[i+3]=130;
            }
          }
        }
      }
    }
    ctx.putImageData(img, 0, 0);

    // Subtle grid (drawn after heatmap so it's visible on top)
    ctx.strokeStyle = isDark ? "rgba(255,255,255,0.06)" : "rgba(0,0,0,0.06)";
    ctx.lineWidth = 1;
    const gs = W / 8;
    for (let gx = gs; gx < W; gx += gs) { ctx.beginPath(); ctx.moveTo(gx,0); ctx.lineTo(gx,H); ctx.stroke(); }
    for (let gy = gs; gy < H; gy += gs) { ctx.beginPath(); ctx.moveTo(0,gy); ctx.lineTo(W,gy); ctx.stroke(); }

    // Margin lines + decision boundary (linear only)
    if (trained && svm.kernel === "linear") {
      const db = svm.decisionBoundary2D();
      const ml = svm.marginLines2D();
      const drawLine = (slope: number, intercept: number, dash: number[], color: string, lw: number) => {
        const x0 = fromPx(0, W), x1 = fromPx(W, W);
        ctx.beginPath();
        ctx.moveTo(toPx(x0, W), toPx(slope*x0 + intercept, H));
        ctx.lineTo(toPx(x1, W), toPx(slope*x1 + intercept, H));
        ctx.strokeStyle = color; ctx.lineWidth = lw;
        ctx.setLineDash(dash); ctx.stroke(); ctx.setLineDash([]);
      };
      if (ml) {
        drawLine(ml.pos.slope, ml.pos.intercept, [5,4], isDark ? "rgba(200,200,200,0.5)" : "rgba(80,80,80,0.4)", 1.5);
        drawLine(ml.neg.slope, ml.neg.intercept, [5,4], isDark ? "rgba(200,200,200,0.5)" : "rgba(80,80,80,0.4)", 1.5);
      }
      if (db) drawLine(db.slope, db.intercept, [], isDark ? "rgba(255,255,255,0.9)" : "rgba(20,20,20,0.85)", 2.5);
    }

    // Points
    const { X, y } = toXy(pts);
    for (let i = 0; i < pts.length; i++) {
      const pt = pts[i];
      const px = toPx(pt.x, W), py = toPx(pt.y, H);
      const [r, g, b] = pt.label === 1 ? COLOR_A : COLOR_B;

      const pointColor = `rgb(${r},${g},${b})`;

      // Support vector ring (linear only)
      if (trained && svm.kernel === "linear" && svm.isSupportVector(X[i], y[i])) {
        ctx.beginPath(); ctx.arc(px, py, 11, 0, Math.PI*2);
        ctx.strokeStyle = isDark ? "rgba(255,255,255,0.55)" : "rgba(30,30,30,0.4)";
        ctx.lineWidth = 2; ctx.stroke();
      }

      // Glow halo
      ctx.shadowBlur = 14;
      ctx.shadowColor = pointColor;

      if (pt.isTest) {
        ctx.fillStyle = pointColor; ctx.strokeStyle = "rgba(255,255,255,0.9)"; ctx.lineWidth = 1.5;
        ctx.fillRect(px-5.5, py-5.5, 11, 11); ctx.strokeRect(px-5.5, py-5.5, 11, 11);
      } else {
        ctx.beginPath(); ctx.arc(px, py, 5.5, 0, Math.PI*2);
        ctx.fillStyle = pointColor; ctx.fill();
        ctx.strokeStyle = "rgba(255,255,255,0.9)"; ctx.lineWidth = 1.5; ctx.stroke();
      }

      ctx.shadowBlur = 0;
    }
  }, [isDark]);

  const runFrame = useCallback((ts: number) => {
    if (!runningRef.current) return;
    const svm = svmRef.current;
    const pts = ptsRef.current;
    const interval = Math.max(16, 200 / (speedScale || 1));
    if (ts - lastFrameRef.current >= interval) {
      lastFrameRef.current = ts;
      const { X, y } = toXy(pts.filter(p => !p.isTest));
      if (X.length >= 2) {
        svm.C = C;
        const loss = svm.fitStep(X, y);
        if (Number.isFinite(loss)) lossHistRef.current.push(loss);
        epochRef.current += 1;
        setEpoch(epochRef.current);
        setAccuracy(svm.accuracy(X, y));
        const mw = svm.marginWidth();
        setMarginStr(isNaN(mw) ? "N/A" : mw.toFixed(3));
        setTrainMetrics(computeMetrics(pts, (x, y2) => svm.predict([x, y2]), false));
        setTestMetrics(computeMetrics(pts, (x, y2) => svm.predict([x, y2]), true));
        drawCanvas();
        renderLoss();
      }
    }
    animRef.current = requestAnimationFrame(runFrame);
  }, [speedScale, C, drawCanvas, renderLoss]);

  const stop = useCallback(() => {
    runningRef.current = false; setIsRunning(false);
    if (animRef.current !== null) { cancelAnimationFrame(animRef.current); animRef.current = null; }
  }, []);

  const start = useCallback(() => {
    if (runningRef.current) return;
    runningRef.current = true; setIsRunning(true);
    animRef.current = requestAnimationFrame(runFrame);
  }, [runFrame]);

  const loadDataset = useCallback((type: DatasetType, n: number, splitEnabled: boolean, kern?: KernelType, gam?: number) => {
    stop();
    const svm = svmRef.current;
    const k   = kern ?? kernel;
    const g   = gam  ?? gamma;
    if (k !== svm.kernel || g !== svm.gamma) svm.setKernel(k, g);
    else svm.reset();
    epochRef.current = 0; setEpoch(0);
    setAccuracy(null); setTestMetrics(null); setTrainMetrics(null); setMarginStr("-");
    lossHistRef.current = [];
    let pts = generateDataset(type, n);
    if (splitEnabled) pts = splitDataset(pts);
    ptsRef.current = pts;
    drawCanvas();
    renderLoss();
    setTimeout(() => start(), 50);
  }, [stop, start, drawCanvas, renderLoss, kernel, gamma]);

  useEffect(() => {
    loadDataset(dataset, noise, trainTestSplit);
    return () => stop();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // R / B keyboard shortcuts — add point at current mouse position
  useEffect(() => {
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
    border: T.brd, boxShadow: `0 1px 3px ${T.sh}`, transition: "all 0.15s",
  });

  const DATASETS: DatasetType[] = ["blobs", "moons", "circles", "xor", "spirals"];

  return (
    <div style={{ display: "flex", gap: 24, alignItems: "flex-start", flexWrap: "wrap", justifyContent: "center",
      fontFamily: "system-ui, -apple-system, sans-serif", color: T.text }}>

      {/* Canvas column */}
      <div style={{ flex: "0 0 auto", display: "flex", flexDirection: "column", gap: 8 }}>
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
            drawCanvas();
          }}
          onContextMenu={(e) => {
            e.preventDefault();
            const rect = (e.target as HTMLCanvasElement).getBoundingClientRect();
            const px = (e.clientX - rect.left) * (CANVAS_W / rect.width);
            const py = (e.clientY - rect.top)  * (CANVAS_H / rect.height);
            ptsRef.current = [...ptsRef.current, { x: fromPx(px, CANVAS_W), y: fromPx(py, CANVAS_H), label: 0 }];
            drawCanvas();
          }}
        />
        {/* Loss curve */}
        <div>
          <div style={{ fontSize: 11, fontWeight: 600, color: T.sub, marginBottom: 3,
            letterSpacing: "0.04em", textTransform: "uppercase" as const }}>Training Loss</div>
          <canvas ref={lossCanvasRef} width={CANVAS_W} height={LOSS_H}
            style={{ border: T.brd, borderRadius: 6, display: "block" }} />
        </div>
        {/* Legend */}
        <div style={{ display: "flex", gap: 14, fontSize: 11, color: T.sub }}>
          <span style={{ color: "#06b6d4" }}>● blue — left-click or B</span>
          <span style={{ color: "#f43f5e" }}>● red — right-click or R</span>
          {kernel === "linear" && <span>◎ support vector</span>}
        </div>
      </div>

      {/* Controls */}
      <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: 16, minWidth: 210 }}>
        {/* Title */}
        <div>
          <h2 style={{ margin: "0 0 6px", fontSize: 22, fontWeight: 700, color: T.text }}>
            Support Vector Machine
          </h2>
          <p style={{ margin: 0, fontSize: 13, color: T.sub, lineHeight: 1.6 }}>
            SGD on hinge loss with soft margin. Linear kernel shows the margin band and
            support vectors. RBF kernel learns non-linear boundaries via random feature maps.
          </p>
        </div>

        {/* Kernel selector */}
        <div>
          <div style={{ fontSize: 12, fontWeight: 600, color: T.text, marginBottom: 6 }}>Kernel</div>
          <div style={{ display: "flex", gap: 6 }}>
            {(["linear", "rbf"] as KernelType[]).map(k => (
              <button key={k} style={{ ...btnStyle(kernel === k), padding: "7px 16px", fontSize: 12 }}
                onClick={() => {
                  setKernel(k);
                  loadDataset(dataset, noise, trainTestSplit, k, gamma);
                }}>
                {k === "linear" ? "Linear" : "RBF"}
              </button>
            ))}
          </div>
        </div>

        {/* Gamma (RBF only) */}
        {kernel === "rbf" && (
          <div>
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, color: T.text, fontWeight: 600, marginBottom: 4 }}>
              <span>γ (bandwidth)</span>
              <span style={{ color: T.sub, fontFamily: "monospace" }}>{gamma.toFixed(1)}</span>
            </div>
            <input type="range" min="0.1" max="10" step="0.1" value={gamma}
              style={{ width: "100%", accentColor: T.acc }}
              onChange={e => {
                const g = Number(e.target.value);
                setGamma(g);
                loadDataset(dataset, noise, trainTestSplit, kernel, g);
              }} />
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, color: T.dimText, marginTop: 2 }}>
              <span>smooth</span><span>complex</span>
            </div>
          </div>
        )}

        {/* Dataset */}
        <div>
          <div style={{ fontSize: 12, fontWeight: 600, color: T.text, marginBottom: 6 }}>Dataset</div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 5 }}>
            {DATASETS.map(ds => (
              <button key={ds} style={btnStyle(dataset === ds)}
                onClick={() => { setDataset(ds); loadDataset(ds, noise, trainTestSplit); }}>
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
          <input type="range" min="0" max="0.3" step="0.01" value={noise}
            style={{ width: "100%", accentColor: T.acc }}
            onChange={e => { const v = Number(e.target.value); setNoise(v); loadDataset(dataset, v, trainTestSplit); }} />
        </div>

        {/* C parameter */}
        <div>
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, color: T.text, fontWeight: 600, marginBottom: 4 }}>
            <span>C (margin penalty)</span><span style={{ color: T.sub, fontFamily: "monospace" }}>{C.toFixed(1)}</span>
          </div>
          <input type="range" min="0.1" max="10" step="0.1" value={C}
            style={{ width: "100%", accentColor: T.acc }}
            onChange={e => setC(Number(e.target.value))} />
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, color: T.dimText, marginTop: 2 }}>
            <span>wide margin</span><span>hard margin</span>
          </div>
        </div>

        {/* Train/test */}
        <label style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 12, color: T.text, cursor: "pointer" }}>
          <input type="checkbox" checked={trainTestSplit} style={{ accentColor: T.acc }}
            onChange={e => { setTrainTestSplit(e.target.checked); loadDataset(dataset, noise, e.target.checked); }} />
          Train/test split (20%)
        </label>

        {/* Buttons */}
        <div style={{ display: "flex", gap: 8 }}>
          <button style={{ padding: "9px 18px", fontSize: 13, fontWeight: 700, borderRadius: 8, border: "none",
            cursor: "pointer", background: isRunning ? "#e53e3e" : T.acc, color: "#fff",
            boxShadow: "0 2px 8px rgba(0,0,0,0.12)", flex: 1 }}
            onClick={() => isRunning ? stop() : start()}>
            {isRunning ? "⏸ Pause" : "▶ Train"}
          </button>
          <button style={{ ...btnStyle(), padding: "9px 18px", fontSize: 13 }}
            onClick={() => loadDataset(dataset, noise, trainTestSplit)}>
            Reset
          </button>
        </div>

        {/* Stats */}
        <div style={{ padding: "12px 14px", background: T.statBg, borderRadius: 10, border: T.brd }}>
          <div style={{ fontSize: 11, fontWeight: 700, color: T.sub, marginBottom: 8, textTransform: "uppercase" as const, letterSpacing: "0.05em" }}>
            SVM Stats
          </div>
          {[
            { label: "Epoch", value: epoch || "—" },
            { label: "Train accuracy", value: fmtPct(accuracy) },
            { label: kernel === "linear" ? "Margin width" : "Kernel", value: kernel === "linear" ? marginStr : `RBF (γ=${gamma.toFixed(1)})` },
          ].map(({ label, value }) => (
            <div key={label} style={{ display: "flex", justifyContent: "space-between", fontSize: 13, marginBottom: 4 }}>
              <span style={{ color: T.sub }}>{label}</span>
              <span style={{ fontWeight: 700, fontFamily: "monospace", color: T.text }}>{String(value)}</span>
            </div>
          ))}
        </div>

        {/* Confusion matrix */}
        {(trainMetrics || testMetrics) && (() => {
          const m = testMetrics ?? trainMetrics;
          if (!m) return null;
          return (
            <div style={{ padding: "12px 14px", background: T.statBg, borderRadius: 10, border: T.brd }}>
              <div style={{ fontSize: 11, fontWeight: 700, color: T.sub, marginBottom: 8, textTransform: "uppercase" as const, letterSpacing: "0.05em" }}>
                {testMetrics ? "Test Metrics" : "Train Metrics"}
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
                <span style={{ color: T.sub }}>Accuracy</span><span style={{ fontWeight: 700, fontFamily: "monospace" }}>{fmtPct(m.acc)}</span>
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
          {kernel === "linear"
            ? <>Minimise ½‖w‖² + C·Σ max(0, 1−yᵢ(w·xᵢ+b))<br />Margin = 2/‖w‖. Higher C → harder, smaller margin.</>
            : <>RBF kernel K(x,z)=exp(−γ‖x−z‖²) approximated via<br />64 random Fourier features, then linear SVM on those features.<br />Higher γ → tighter, more complex boundary.</>
          }
        </div>

        <div style={{ fontSize: 11, color: T.dimText, lineHeight: 1.6 }}>
          <strong>Tip:</strong> {kernel === "linear"
            ? "Linear SVM works best on linearly separable data. Try Blobs, then switch to RBF for moons or circles."
            : "RBF can fit any dataset — watch out for overfitting at high γ and low C."}
        </div>
      </div>
    </div>
  );
};

export default SvmDemo;
