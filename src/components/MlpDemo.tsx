import React, { useEffect, useRef, useState } from "react";
import p5 from "p5";
import MLPClassifier from "../utils/mlp";
import type { P5Instance } from "../types/p5";

type Point = { x: number; y: number; label?: string };

type MlpDemoProps = {
  showInstructions?: boolean;
  speedScale?: number;
  dataset?: Point[];
  onDatasetChange?: (updater: ((d: Point[]) => Point[]) | Point[]) => void;
  resetToken?: any;
};

const MlpDemo: React.FC<MlpDemoProps> = ({ showInstructions = true, speedScale = 1, dataset: externalDataset, onDatasetChange, resetToken }) => {
  const sketchRef = useRef<HTMLDivElement | null>(null);
  const p5Ref = useRef<P5Instance | null>(null);
  const [loss, setLoss] = useState<number | null>(null);
  // control panel state
  const [optimizer, setOptimizer] = useState<string>("sgd");
  const [lr, setLr] = useState<number>(0.3);
  const [batchSize, setBatchSize] = useState<number>(8);
  const [epochs, setEpochs] = useState<number>(20);
  const [hiddenUnits, setHiddenUnits] = useState<number>(16);
  const [gridStepState, setGridStepState] = useState<number>(8);
  // touch support: which class a tap should add on touch devices
  const [touchClass, setTouchClass] = useState<"A" | "B">("A");
  // draggable panel state
  const panelRef = useRef<HTMLDivElement | null>(null);
  // panel is fixed (non-draggable) and placed above the keyboard help box by default
  const [panelPos, setPanelPos] = useState<{ x: number; y: number } | null>(null);
  const [panelBottomOffset, setPanelBottomOffset] = useState<number>(140);
  // compute color scheme for controls (match system preference)
  const prefersDark = typeof window !== "undefined" && window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
  const panelBg = prefersDark ? "rgba(45,55,72,0.95)" : "rgba(255,255,255,0.95)";
  const panelText = prefersDark ? "#f7fafc" : "#1a202c";
  const panelBorder = prefersDark ? "1px solid rgba(255,255,255,0.06)" : "1px solid rgba(0,0,0,0.06)";
  const inputBg = prefersDark ? "#2d3748" : "#ffffff";
  const inputColor = panelText;
  const controlShadow = prefersDark ? "0 8px 32px rgba(0,0,0,0.6)" : "0 8px 32px rgba(0,0,0,0.08)";

  useEffect(() => {
    const sketch = (p: P5Instance) => {
      let points: Point[] = [];
      let X: number[][] = [];
      let y: number[] = [];
      let model: MLPClassifier | null = null;
  let gridG: any = null;
  const defaultGridStep = gridStepState;
      let gridStep = defaultGridStep;
      let gridFreq = 6;
      let gridDirty = true;

      const resetModel = (hidden: number[] = [16]) => {
        model = new MLPClassifier(2, hidden, { lr: lr, optimizer: optimizer });
      };

      const resetDemo = () => {
        points = [];
        X = [];
        y = [];
        for (let i = 0; i < 50; i++) {
          const vx = p.random(-1, 1);
          const vy = p.random(-1, 1);
          const label = vx * 0.4 + 0.1 > vy ? "A" : "B";
          points.push({ x: vx, y: vy, label });
          X.push([vx, vy]);
          y.push(label === "A" ? 1 : 0);
        }
        resetModel();
        gridDirty = true;
      };

      p.setup = () => {
        if (sketchRef.current) sketchRef.current.querySelectorAll("canvas").forEach((c) => c.remove());
        const rect = sketchRef.current?.getBoundingClientRect();
        p.createCanvas(Math.max(300, rect?.width || 600), Math.max(300, rect?.height || 600));
        gridG = p.createGraphics(p.width, p.height);
        resetDemo();
        if (externalDataset && Array.isArray(externalDataset) && externalDataset.length) p.updateDataset(externalDataset);

        // attach pointer handler directly to the canvas for responsive adding
        try {
          const root = sketchRef.current;
          if (root) {
            const canvas = root.querySelector("canvas");
            if (canvas) {
              const pointerHandler = (ev: PointerEvent) => {
                const rect = canvas.getBoundingClientRect();
                const px = ev.clientX - rect.left;
                const py = ev.clientY - rect.top;
                if (px < 0 || py < 0 || px > p.width || py > p.height) return;
                const mx = p.map(px, 0, p.width, -1, 1);
                const my = p.map(py, 0, p.height, 1, -1);
                // choose label: on touch use selected touchClass; otherwise mouse button
                let label = "A";
                try {
                  if ((ev as any).pointerType === "touch") {
                    label = (p as any)._mlpControls?.touchClass ?? "A";
                  } else {
                    label = ev.button === 2 ? "B" : "A";
                  }
                } catch (e) {
                  label = ev.button === 2 ? "B" : "A";
                }
                points.push({ x: mx, y: my, label });
                X.push([mx, my]);
                y.push(label === "A" ? 1 : 0);
                gridDirty = true;
                if (onDatasetChange) onDatasetChange((d: Point[] = []) => [...d, { x: mx, y: my, label }]);
              };
              const ctxHandler = (e: Event) => e.preventDefault();
              canvas.addEventListener("pointerdown", pointerHandler);
              canvas.addEventListener("contextmenu", ctxHandler);
              (canvas as any)._mlpPointerHandler = pointerHandler;
              (canvas as any)._mlpContextHandler = ctxHandler;
            }
          }
        } catch (e) {}
        // expose control values on the p5 instance so React controls can update the running sketch
        try {
          (p as any)._mlpControls = {
            get optimizer() { return optimizer; },
            get lr() { return lr; },
            get batchSize() { return batchSize; },
            get epochs() { return epochs; },
            get hiddenUnits() { return hiddenUnits; },
            get touchClass() { return touchClass; },
            setGridStep: (v: number) => { gridStep = v; gridDirty = true; },
          };
        } catch (e) {}
      };

      p.draw = () => {
        // take a few random sample gradient steps per frame so UI stays responsive
        if (model && X.length) {
          const steps = Math.max(1, Math.floor(1 * (p.speedScale || 1)));
          for (let s = 0; s < steps; s++) {
            const i = Math.floor(Math.random() * X.length);
            model.trainSample(X[i], y[i], 0.3);
          }
        }

        const darkMode = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
        p.background(darkMode ? 30 : 255);

        if (!gridG) gridG = p.createGraphics(p.width, p.height);
        if (p.frameCount % gridFreq === 0 || gridDirty) {
          gridDirty = false;
          gridG.clear();
          for (let gx = 0; gx < p.width; gx += gridStep) {
            for (let gy = 0; gy < p.height; gy += gridStep) {
              const nx = p.map(gx + 0.5 * gridStep, 0, p.width, -1, 1);
              const ny = p.map(gy + 0.5 * gridStep, p.height, 0, -1, 1);
              const pred = model ? (model.forward([nx, ny]) >= 0.5 ? 1 : 0) : 0;
              const redRGBA = darkMode ? "rgba(229,62,62,0.25)" : "rgba(229,62,62,0.12)";
              const blueRGBA = darkMode ? "rgba(66,153,225,0.25)" : "rgba(66,153,225,0.12)";
              gridG.noStroke();
              gridG.fill(pred === 1 ? (redRGBA as any) : (blueRGBA as any));
              gridG.rect(gx, gy, gridStep, gridStep);
            }
          }
        }
        p.image(gridG, 0, 0);

        for (const pt of points) {
          const px = p.map(pt.x, -1, 1, 0, p.width);
          const py = p.map(pt.y, -1, 1, p.height, 0);
          p.fill(pt.label === "A" ? "#e53e3e" : "#4299e1");
          p.noStroke();
          p.circle(px, py, 12);
        }
      };

      p.mousePressed = () => {
        if (!p5Ref.current) return;
        if (p.mouseX < 0 || p.mouseY < 0 || p.mouseX > p.width || p.mouseY > p.height) return;
        const mx = p.map(p.mouseX, 0, p.width, -1, 1);
        const my = p.map(p.mouseY, p.height, 0, -1, 1);
        // on touch devices use selected touchClass; p.touches exists when touching
        let label = "A";
        try {
          const isTouch = (p as any).touches && (p as any).touches.length > 0;
          if (isTouch) {
            label = (p as any)._mlpControls?.touchClass ?? "A";
          } else {
            label = p.mouseButton === p.LEFT ? "A" : "B";
          }
        } catch (e) {
          label = p.mouseButton === p.LEFT ? "A" : "B";
        }
        points.push({ x: mx, y: my, label });
        X.push([mx, my]);
        y.push(label === "A" ? 1 : 0);
        gridDirty = true;
        if (onDatasetChange) onDatasetChange((d: Point[] = []) => [...d, { x: mx, y: my, label }]);
      };

      p.updateDataset = (newDataset: Point[] | undefined) => {
        points = [];
        X = [];
        y = [];
        if (newDataset && Array.isArray(newDataset)) {
          for (const pt of newDataset) {
            points.push({ x: pt.x, y: pt.y, label: pt.label });
            X.push([pt.x, pt.y]);
            y.push(pt.label === "A" ? 1 : 0);
          }
        }
        resetModel();
        gridDirty = true;
      };

      p.trainMLP = (opts: { epochs?: number; lr?: number } = {}) => {
        if (!model) resetModel();
        if (!model || !X.length) return;
        const epochs = opts.epochs ?? Math.max(5, Math.floor(20 / Math.max(1, speedScale)));
        const lr = opts.lr ?? 0.3;
        model.fit(X, y, {
          epochs,
          lr,
          batchSize: 8,
          shuffle: true,
          onEpoch: (epoch: number, L: number) => {
            // store last loss on p so React can read it
            try {
              (p as any)._lastLoss = L;
            } catch (e) {}
          },
        });
        gridDirty = true;
      };

      p.resetDemo = () => {
        resetDemo();
        if (onDatasetChange) onDatasetChange([]);
      };
    };

    if (p5Ref.current) {
      p5Ref.current.remove();
      p5Ref.current = null;
    }
    p5Ref.current = new p5(sketch, sketchRef.current as Element);

    return () => {
      if (sketchRef.current) {
        const canvas = sketchRef.current.querySelector("canvas");
        if (canvas) {
          const ph = (canvas as any)._mlpPointerHandler;
          const ch = (canvas as any)._mlpContextHandler;
          if (ph) canvas.removeEventListener("pointerdown", ph);
          if (ch) canvas.removeEventListener("contextmenu", ch);
        }
      }
      if (p5Ref.current) {
        p5Ref.current.remove();
        p5Ref.current = null;
      }
    };
  }, [externalDataset, onDatasetChange, speedScale]);

  useEffect(() => {
    if (p5Ref.current && typeof p5Ref.current.updateDataset === "function") {
      p5Ref.current.updateDataset(externalDataset || []);
    }
  }, [externalDataset]);

  // poll p5 instance for lastLoss set by training callback and update React state
  useEffect(() => {
    const int = setInterval(() => {
      if (p5Ref.current) {
        const L = (p5Ref.current as any)._lastLoss;
        if (typeof L === "number") setLoss(L);
      }
    }, 200);
    return () => clearInterval(int);
  }, []);

  // No automatic panelPos initialization — the panel is fixed outside the canvas
  // (right:20, bottom:140) and is not draggable.

  // Fixed bottom offset — help box was moved to the top-right so we no longer
  // need to dynamically compute an offset to avoid overlap. Keep a comfortable
  // distance from the bottom to avoid touching system UI on mobile.
  useEffect(() => {
    setPanelBottomOffset(140);
  }, []);

  // pointer move/up handlers for dragging the panel
  // panel is not draggable; no pointer listeners required

  useEffect(() => {
    if (p5Ref.current && typeof p5Ref.current.trainMLP === "function") {
      setTimeout(() => p5Ref.current.trainMLP({ epochs: Math.max(8, Math.floor(20 / Math.max(1, speedScale))) }), 50);
    }
  }, [resetToken, speedScale]);

  return (
  <div ref={sketchRef} style={{ position: "relative", width: "100%", height: "100%", overflow: "hidden" }}>
      <div style={{ position: "absolute", left: 12, top: 12, zIndex: 20 }}>
        <button
          onClick={() => p5Ref.current && typeof p5Ref.current.resetDemo === "function" && p5Ref.current.resetDemo()}
          style={{ marginRight: 8 }}
        >
          Reset
        </button>
        <button
          onClick={() => p5Ref.current && typeof p5Ref.current.trainMLP === "function" && p5Ref.current.trainMLP({ epochs: 20 })}
        >
          Train
        </button>
      </div>
      {showInstructions && (
        <div style={{ position: "absolute", right: 12, top: 12, zIndex: 20, maxWidth: 280 }}>
          <div style={{ padding: 8, background: "rgba(255,255,255,0.9)", borderRadius: 8 }}>
            <strong>MLP demo</strong>
            <div style={{ fontSize: 12 }}>Click to add points (left=class A, right=class B).</div>
          </div>
        </div>
      )}
      {/* control panel */}
      <div
        ref={panelRef}
        style={{ position: "fixed", right: 20, bottom: panelBottomOffset, zIndex: 1150, touchAction: 'none', cursor: 'default' }}
      >
        <div style={{ padding: 8, background: panelBg, borderRadius: 8, width: 220, boxSizing: 'border-box', color: panelText, border: panelBorder, boxShadow: controlShadow }}>
          <div style={{ marginBottom: 6 }}><strong>Model Controls</strong></div>
          <div style={{ fontSize: 12, marginBottom: 6 }}>
            <label style={{ color: panelText }}>Optimizer: </label>
            <select value={optimizer} onChange={(e) => { setOptimizer(e.target.value); if (p5Ref.current) (p5Ref.current as any)._mlpControls.optimizer = e.target.value; }} style={{ marginLeft: 6, background: inputBg, color: inputColor, borderRadius: 6, border: panelBorder }}>
              <option value="sgd">SGD</option>
              <option value="momentum">Momentum</option>
              <option value="adam">Adam</option>
            </select>
          </div>
          <div style={{ fontSize: 12, marginBottom: 6 }}>
            <label style={{ color: panelText }}>LR: </label>
            <input type="number" step="0.01" value={lr} onChange={(e) => { const v = Number(e.target.value); setLr(v); if (p5Ref.current) (p5Ref.current as any)._mlpControls.lr = v; }} style={{ width: 70, marginLeft: 6, background: inputBg, color: inputColor, borderRadius: 6, border: panelBorder }} />
          </div>
          <div style={{ fontSize: 12, marginBottom: 6 }}>
            <label style={{ color: panelText }}>Batch: </label>
            <input type="number" value={batchSize} onChange={(e) => { const v = Number(e.target.value); setBatchSize(v); if (p5Ref.current) (p5Ref.current as any)._mlpControls.batchSize = v; }} style={{ width: 70, marginLeft: 6, background: inputBg, color: inputColor, borderRadius: 6, border: panelBorder }} />
          </div>
          <div style={{ fontSize: 12, marginBottom: 6 }}>
            <label style={{ color: panelText }}>Epochs: </label>
            <input type="number" value={epochs} onChange={(e) => { const v = Number(e.target.value); setEpochs(v); if (p5Ref.current) (p5Ref.current as any)._mlpControls.epochs = v; }} style={{ width: 70, marginLeft: 6, background: inputBg, color: inputColor, borderRadius: 6, border: panelBorder }} />
          </div>
          <div style={{ fontSize: 12, marginBottom: 6 }}>
            <label style={{ color: panelText }}>Hidden: </label>
            <input type="number" value={hiddenUnits} onChange={(e) => { const v = Number(e.target.value); setHiddenUnits(v); if (p5Ref.current) (p5Ref.current as any)._mlpControls.hiddenUnits = v; }} style={{ width: 70, marginLeft: 6, background: inputBg, color: inputColor, borderRadius: 6, border: panelBorder }} />
          </div>
          <div style={{ fontSize: 12, marginBottom: 6 }}>
            <label style={{ color: panelText }}>Grid: </label>
            <input type="number" value={gridStepState} onChange={(e) => { const v = Number(e.target.value); setGridStepState(v); if (p5Ref.current) (p5Ref.current as any)._mlpControls.setGridStep?.(v); }} style={{ width: 70, marginLeft: 6, background: inputBg, color: inputColor, borderRadius: 6, border: panelBorder }} />
          </div>
          <div style={{ fontSize: 12, marginBottom: 6 }}>
            <label style={{ color: panelText }}>Touch tap adds: </label>
            <select value={touchClass} onChange={(e) => { const v = e.target.value as "A" | "B"; setTouchClass(v); if (p5Ref.current) (p5Ref.current as any)._mlpControls.touchClass = v; }} style={{ marginLeft: 6, background: inputBg, color: inputColor, borderRadius: 6, border: panelBorder }}>
              <option value="A">Red (A)</option>
              <option value="B">Blue (B)</option>
            </select>
          </div>
        </div>
      </div>
      {/* live loss overlay */}
      <div style={{ position: "absolute", left: 12, bottom: 12, zIndex: 30, padding: 8, background: "rgba(0,0,0,0.6)", color: "#fff", borderRadius: 6, fontSize: 12 }}>
        Loss: {loss == null ? '—' : loss.toFixed(4)}
      </div>
    </div>
  );
};

export default MlpDemo;
