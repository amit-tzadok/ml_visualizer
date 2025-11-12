import React, { useEffect, useRef, useState } from "react";
// The MLP sketch exposes a number of control getters/setters on the p5 instance
// which intentionally read from refs to avoid rebuilding the heavy sketch when UI controls change.
import MLPClassifier, { ActivationFunction } from "../utils/mlp";
import loadP5 from "../utils/loadP5";
import type { P5Instance } from "../types/p5";
import type { P5Graphics } from "../types";

type Point = { x: number; y: number; label?: string };

type MlpDemoProps = {
  showInstructions?: boolean;
  speedScale?: number;
  dataset?: Point[];
  onDatasetChange?: (updater: ((d: Point[]) => Point[]) | Point[]) => void;
  resetToken?: unknown;
  compact?: boolean;
};

const MlpDemo: React.FC<MlpDemoProps> = ({
  showInstructions = true,
  speedScale = 1,
  dataset: externalDataset,
  onDatasetChange,
  resetToken,
  compact = false,
}) => {
  const sketchRef = useRef<HTMLDivElement | null>(null);
  const p5Ref = useRef<P5Instance | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [loss, setLoss] = useState<number | null>(null);
  const finishedRef = useRef<HTMLDivElement | null>(null);
  const [mlpMeanPred, setMlpMeanPred] = useState<number | null>(null);
  const [mlpAcc, setMlpAcc] = useState<number | null>(null);
  // timing overlay for training duration
  const [_lastDurationSec, setLastDurationSec] = useState<number | null>(null);
  // control panel state
  // Revert to SGD by default for stability in the interactive demo
  const [optimizer, setOptimizer] = useState<string>("sgd");
  const [activation, setActivation] = useState<string>("sigmoid");
  // Default lr tuned for SGD
  const [lr, setLr] = useState<number>(0.3);
  const [batchSize, setBatchSize] = useState<number>(8);
  const [epochs, setEpochs] = useState<number>(100);
  const [hiddenUnits, setHiddenUnits] = useState<number>(32);
  // Legacy/Stable toggle: when enabled, the demo will use conservative, proven defaults
  // (single hidden layer, 16 units, SGD @ 0.3, sigmoid). This allows switching back
  // to the previous working behavior without reverting algorithmic changes.
  const [legacyStable, setLegacyStable] = useState<boolean>(false);
  const [gridStepState, setGridStepState] = useState<number>(4);
  // touch support: which class a tap should add on touch devices
  const [touchClass, setTouchClass] = useState<"A" | "B">("A");
  // draggable panel state
  const panelRef = useRef<HTMLDivElement | null>(null);
  // controlsRef intentionally starts null to avoid closing over React state during render.
  // The sketch will read values via getters that fallback to stable literals until this ref is populated.
  const controlsRef = useRef<{
    optimizer: string;
    activation: string;
    lr: number;
    batchSize: number;
    epochs: number;
    hiddenUnits: number;
    touchClass: "A" | "B";
    setGridStep: (v: number) => void;
  } | null>(null);
  // panel is fixed (non-draggable) and placed above the keyboard help box by default
  const [_panelPos, _setPanelPos] = useState<{ x: number; y: number } | null>(
    null
  );
  const [_panelBottomOffset, setPanelBottomOffset] = useState<number>(140);
  // compute color scheme for controls (match system preference)
  const prefersDark =
    typeof window !== "undefined" &&
    window.matchMedia &&
    window.matchMedia("(prefers-color-scheme: dark)").matches;
  const panelBg = prefersDark
    ? "rgba(45,55,72,0.95)"
    : "rgba(255,255,255,0.95)";
  const panelText = prefersDark ? "#f7fafc" : "#1a202c";
  const panelBorder = prefersDark
    ? "1px solid rgba(255,255,255,0.06)"
    : "1px solid rgba(0,0,0,0.06)";
  const inputBg = prefersDark ? "#2d3748" : "#ffffff";
  const inputColor = panelText;
  const controlShadow = prefersDark
    ? "0 8px 32px rgba(0,0,0,0.6)"
    : "0 8px 32px rgba(0,0,0,0.08)";
  // runtime helper to detect dev environment without using `any` casts
  const isDev = () => {
    try {
      const proc = (
        globalThis as unknown as { process?: { env?: { NODE_ENV?: string } } }
      ).process;
      return !!(proc && proc.env && proc.env.NODE_ENV !== "production");
    } catch {
      return false;
    }
  };

  useEffect(() => {
    const sketch = (p: P5Instance) => {
      type P5MlpExt = {
        _setGridStep?: (v: number) => void;
        _mlpControls?: P5Instance["_mlpControls"];
        [k: string]: unknown;
      };
      const pEx = p as unknown as P5Instance & P5MlpExt;
      let points: Point[] = [];
      let X: number[][] = [];
      let y: number[] = [];
      let model: MLPClassifier | null = null;
      let gridG: P5Graphics | null = null;
      // Only train during an explicit training run
      let isTraining: boolean = false;
      let trainingEpochsRemaining = 0;
      let trainingLR = 0.3;
      let samplesInCurrentEpoch = 0;
      // Track training start time for elapsed duration
      let startedAtMs: number | null = null;
      // Adaptive grid step with optional user override via control panel
      let overrideStep: number | null = null;
      // Start with a small default; computeAdaptiveStep will recalculate based on canvas size.
      let adaptiveStep = 4;
      let gridStep = adaptiveStep;
      let gridFreq = 18; // update less often for performance
      let gridDirty = true;
  // smoothing buffer to avoid global color flips when the model changes quickly
  let prevPreds: Float32Array | null = null;
  let prevCols = 0;
  let prevRows = 0;
  const PRED_SMOOTH = 0.15; // new -> prev blending factor (small = smoother)

      let _suppressStatusUntil = 0;

      // allow external callers to suppress publishing runtime status for a short window
      // (used after Reset to avoid immediately repopulating overlays)
      (p as unknown as { suppressStatus?: (ms: number) => void }).suppressStatus = (ms: number) => {
        try {
          _suppressStatusUntil = Date.now() + Math.max(0, Number(ms) || 0);
        } catch (err) {
          if (isDev()) console.debug('mlp: prevPreds reset error', err);
        }
      };

      const resetModel = (hidden: number[] = []) => {
        // Read control values from controlsRef to avoid closing over React state in the sketch.
        const ctrl = controlsRef.current;
        const lrVal = ctrl?.lr ?? 0.3;
        const opt = ctrl?.optimizer ?? "sgd";
        const act = (ctrl?.activation ?? "sigmoid") as ActivationFunction;
        const hiddenUnitsVal = ctrl?.hiddenUnits ?? 16;
        // Use two hidden layers by default for higher capacity in the demo:
        // primary layer = hiddenUnitsVal, secondary = ~half (min 8)
        const h = hidden.length
          ? hidden
          : [hiddenUnitsVal, Math.max(8, Math.floor(hiddenUnitsVal / 2))];
        model = new MLPClassifier(2, h, {
          lr: lrVal,
          optimizer: opt,
          activation: act,
        });
        // reset smoothing buffer when model is recreated
  try { prevPreds = null; prevCols = 0; prevRows = 0; } catch (err) { if (isDev()) console.debug('mlp: prevPreds reset error', err); }
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

      const computeAdaptiveStep = () => {
        const minDim = Math.max(1, Math.min(p.width || 600, p.height || 600));
        // Aim for ~150-200 cells on the short edge for smoother boundaries
        const base = Math.max(3, Math.floor(minDim / 150));
        adaptiveStep = base;
      };

      p.setup = () => {
        if (sketchRef.current)
          sketchRef.current
            .querySelectorAll("canvas")
            .forEach((c) => c.remove());
        const rect = sketchRef.current?.getBoundingClientRect();
        p.createCanvas(
          Math.max(300, rect?.width || 600),
          Math.max(300, rect?.height || 600)
        );
        p.frameRate?.(45);
        if (typeof p.pixelDensity === "function") p.pixelDensity(1);
        gridG = p.createGraphics
          ? (p.createGraphics(p.width, p.height) as P5Graphics)
          : null;
        computeAdaptiveStep();
        gridStep = overrideStep ?? adaptiveStep;
        resetDemo();
        if (
          externalDataset &&
          Array.isArray(externalDataset) &&
          externalDataset.length
        )
          p.updateDataset(externalDataset);

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
                  if (
                    (ev as PointerEvent & { pointerType?: string })
                      .pointerType === "touch"
                  ) {
                    label = p._mlpControls?.touchClass ?? "A";
                  } else {
                    label = ev.button === 2 ? "B" : "A";
                  }
                } catch (err) {
                  // Surface touch detection problems during development but keep production quiet
                  if (isDev()) console.debug("mlp: touch detection error", err);
                  label = ev.button === 2 ? "B" : "A";
                }
                points.push({ x: mx, y: my, label });
                X.push([mx, my]);
                y.push(label === "A" ? 1 : 0);
                gridDirty = true;
                if (onDatasetChange)
                  onDatasetChange((d: Point[] = []) => [
                    ...d,
                    { x: mx, y: my, label },
                  ]);
              };
              const ctxHandler = (e: Event) => e.preventDefault();
              canvas.addEventListener("pointerdown", pointerHandler);
              canvas.addEventListener("contextmenu", ctxHandler);
              const canvasExt = canvas as unknown as {
                _mlpPointerHandler?: (e: Event) => void;
                _mlpContextHandler?: (e: Event) => void;
              };
              canvasExt._mlpPointerHandler = pointerHandler;
              canvasExt._mlpContextHandler = ctxHandler;
              // No automatic retrain on pointer up (reverted)
            }
          }
        } catch (err) {
          // Surface canvas setup problems during development but keep production quiet
          if (isDev()) console.debug("mlp: canvas setup error", err);
        }
        // expose control values on the p5 instance via a ref-backed proxy so we don't close over
        // React state values (avoids rebuilding the sketch when controls change)
        try {
          // expose a setter that mutates the local overrideStep/gridDirty inside the sketch
          pEx._setGridStep = (v: number) => {
            overrideStep = Math.max(2, v);
            gridStep = overrideStep;
            gridDirty = true;
            // reset smoothing buffer when grid resolution changes
            try { prevPreds = null; prevCols = 0; prevRows = 0; } catch (err) { if (isDev()) console.debug('mlp: prevPreds reset error', err); }
          };
          p._mlpControls = {
            // Use ref-backed values; fall back to stable literals so the sketch-creation effect
            // doesn't close over React state variables (avoids rebuilds and satisfies exhaustive-deps).
            get optimizer() {
              return controlsRef.current?.optimizer ?? "sgd";
            },
            set optimizer(v: string) {
              try {
                if (!controlsRef.current)
                  controlsRef.current = {
                    optimizer: v,
                    activation: "sigmoid",
                    lr: 0.3,
                    batchSize: 8,
                    epochs: 100,
                    hiddenUnits: 16,
                    touchClass: "A",
                    setGridStep: controlsRef.current?.setGridStep ?? (() => {}),
                  } as unknown as NonNullable<typeof controlsRef.current>;
                else controlsRef.current.optimizer = v;
                gridDirty = true;
              } catch (err) {
                if (isDev()) console.debug('mlp: control init error', err);
              }
            },
            get activation() {
              return controlsRef.current?.activation ?? "sigmoid";
            },
            set activation(v: string) {
              try {
                if (!controlsRef.current)
                  controlsRef.current = {
                    optimizer: "sgd",
                    activation: v,
                    lr: 0.3,
                    batchSize: 8,
                    epochs: 100,
                    hiddenUnits: 16,
                    touchClass: "A",
                    setGridStep: controlsRef.current?.setGridStep ?? (() => {}),
                  } as unknown as NonNullable<typeof controlsRef.current>;
                else controlsRef.current.activation = v;
                gridDirty = true;
              } catch (err) {
                if (isDev()) console.debug('mlp: fullFit finalAccuracy error', err);
              }
            },
            get lr() {
              return controlsRef.current?.lr ?? 0.3;
            },
            set lr(v: number) {
              try {
                if (!controlsRef.current)
                  controlsRef.current = {
                    optimizer: "sgd",
                    activation: "sigmoid",
                    lr: v,
                    batchSize: 8,
                    epochs: 100,
                    hiddenUnits: 16,
                    touchClass: "A",
                    setGridStep: controlsRef.current?.setGridStep ?? (() => {}),
                  } as unknown as NonNullable<typeof controlsRef.current>;
                else controlsRef.current.lr = v;
              } catch (err) {
                if (isDev()) console.debug('mlp: lr setter error', err);
              }
            },
            get batchSize() {
              return controlsRef.current?.batchSize ?? 8;
            },
            set batchSize(v: number) {
              try {
                if (!controlsRef.current)
                  controlsRef.current = {
                    optimizer: "sgd",
                    activation: "sigmoid",
                    lr: 0.3,
                    batchSize: v,
                    epochs: 100,
                    hiddenUnits: 16,
                    touchClass: "A",
                    setGridStep: controlsRef.current?.setGridStep ?? (() => {}),
                  } as unknown as NonNullable<typeof controlsRef.current>;
                else controlsRef.current.batchSize = v;
              } catch (err) {
                if (isDev()) console.debug('mlp: batchSize setter error', err);
              }
            },
            get epochs() {
              return controlsRef.current?.epochs ?? 100;
            },
            set epochs(v: number) {
              try {
                if (!controlsRef.current)
                  controlsRef.current = {
                    optimizer: "sgd",
                    activation: "sigmoid",
                    lr: 0.3,
                    batchSize: 8,
                    epochs: v,
                    hiddenUnits: 16,
                    touchClass: "A",
                    setGridStep: controlsRef.current?.setGridStep ?? (() => {}),
                  } as unknown as NonNullable<typeof controlsRef.current>;
                else controlsRef.current.epochs = v;
              } catch (err) {
                if (isDev()) console.debug('mlp: epochs setter error', err);
              }
            },
            get hiddenUnits() {
              return controlsRef.current?.hiddenUnits ?? 16;
            },
            set hiddenUnits(v: number) {
              try {
                if (!controlsRef.current)
                  controlsRef.current = {
                    optimizer: "sgd",
                    activation: "sigmoid",
                    lr: 0.3,
                    batchSize: 8,
                    epochs: 100,
                    hiddenUnits: v,
                    touchClass: "A",
                    setGridStep: controlsRef.current?.setGridStep ?? (() => {}),
                  } as unknown as NonNullable<typeof controlsRef.current>;
                else controlsRef.current.hiddenUnits = v;
                gridDirty = true;
              } catch (err) {
                if (isDev()) console.debug('mlp: hiddenUnits setter error', err);
              }
            },
            get touchClass() {
              return controlsRef.current?.touchClass ?? "A";
            },
            set touchClass(v: "A" | "B") {
              try {
                if (!controlsRef.current)
                  controlsRef.current = {
                    optimizer: "sgd",
                    activation: "sigmoid",
                    lr: 0.3,
                    batchSize: 8,
                    epochs: 100,
                    hiddenUnits: 16,
                    touchClass: v,
                    setGridStep: controlsRef.current?.setGridStep ?? (() => {}),
                  } as unknown as NonNullable<typeof controlsRef.current>;
                else controlsRef.current.touchClass = v;
              } catch (err) {
                if (isDev()) console.debug('mlp: touchClass setter error', err);
              }
            },
            setGridStep: (v: number) => {
              try {
                pEx._setGridStep?.(v);
              } catch (err) {
                if (isDev()) console.debug("mlp: setGridStep error", err);
              }
            },
          } as unknown as P5Instance["_mlpControls"];
        } catch (err) {
          // Surface control-attachment problems during development but keep production quiet
          if (isDev()) console.debug("mlp: control setup error", err);
        }
      };

      pEx.windowResized = () => {
        const rect = sketchRef.current?.getBoundingClientRect();
        p.resizeCanvas?.(
          Math.max(300, rect?.width || 600),
          Math.max(300, rect?.height || 600)
        );
        gridG = p.createGraphics
          ? (p.createGraphics(p.width, p.height) as P5Graphics)
          : null;
        computeAdaptiveStep();
        gridStep = overrideStep ?? adaptiveStep;
        gridDirty = true;
      };

      p.draw = () => {
        // Skip heavy updates if tab is hidden to save resources
        try {
          if (
            typeof document !== "undefined" &&
            (document as Document & { hidden?: boolean }).hidden
          ) {
            return;
          }
        } catch (err) {
          if (isDev()) console.debug("mlp: visibility probe error", err);
        }
        // Only perform per-frame updates if inside an active training run
        if (isTraining && model && X.length && trainingEpochsRemaining > 0) {
          const steps = Math.max(
            3,
            Math.min(10, Math.floor(5 * (p.speedScale || 1)))
          );
          for (let s = 0; s < steps; s++) {
            const i = Math.floor(Math.random() * X.length);
            model.trainSample(X[i], y[i], trainingLR);
            samplesInCurrentEpoch++;

            // Complete epoch after training on all samples
            if (samplesInCurrentEpoch >= X.length) {
              samplesInCurrentEpoch = 0;
              trainingEpochsRemaining--;

              // Calculate and store loss
              try {
                const L = model.loss(X, y);
                p._lastLoss = L;
              } catch (err) {
                if (isDev()) console.debug("mlp: model.loss error", err);
              }

              gridDirty = true;

              // Check if training is complete
              if (trainingEpochsRemaining <= 0) {
                isTraining = false;
                // Calculate accuracy
                let accuracy = 0;
                try {
                  if (model && typeof model.accuracy === "function") {
                    accuracy = model.accuracy(X, y);
                  }
                } catch (err) {
                  if (isDev()) console.debug("mlp: accuracy calc error", err);
                }
                // Compute elapsed time
                let elapsedSec: number | null = null;
                try {
                  const now =
                    typeof performance !== "undefined" &&
                    typeof performance.now === "function"
                      ? performance.now()
                      : Date.now();
                  if (startedAtMs != null)
                    elapsedSec = (now - startedAtMs) / 1000;
                } catch (err) {
                  if (isDev())
                    console.debug("mlp: time measurement error", err);
                }
                // Notify completion
                try {
                  window.dispatchEvent(
                    new CustomEvent("mlv:demo-finished", {
                      detail: {
                        classifier: "Neural Network (MLP)",
                        reason: "train-complete",
                        accuracy,
                        elapsedSec,
                      },
                    })
                  );
                } catch (err) {
                  if (isDev()) console.debug("mlp: event dispatch error", err);
                }
                // Show finished overlay
                try {
                  if (finishedRef.current) {
                    const L = p._lastLoss;
                    const timeStr =
                      elapsedSec != null
                        ? ` • Time: ${elapsedSec.toFixed(2)}s`
                        : "";
                    const msg =
                      typeof L === "number"
                        ? `Training finished — Loss: ${L.toFixed(4)}${timeStr}`
                        : `Training finished${timeStr}`;
                    finishedRef.current.innerText = msg;
                    finishedRef.current.style.display = "block";
                  }
                } catch (err) {
                  if (isDev()) console.debug("mlp: DOM update error", err);
                }
              }
            }
          }
        }

        const isDark =
          window.matchMedia &&
          window.matchMedia("(prefers-color-scheme: dark)").matches;
        // Darken canvas a bit (light 250 -> 240, dark 48 -> 42) for improved contrast
        p.background(isDark ? 42 : 240);

        // Publish status for AgentPanel (MLP has no simple closed-form equation)
        try {
          let status: Record<string, unknown> = {
            classifier: "mlp",
            equation: null,
            weights: undefined,
            bias: undefined,
            updatedAt: Date.now(),
          };
          // honor suppression window
          if (Date.now() < _suppressStatusUntil) {
            status.suppressed = true;
          } else {
            // If model exists expose a couple of runtime diagnostics to help debug
            if (model && X.length) {
            try {
              let sumPred = 0;
              let ok = 0;
              for (let i = 0; i < X.length; i++) {
                const pval = model.forward(X[i]);
                sumPred += pval;
                const cl = pval >= 0.5 ? 1 : 0;
                if (cl === y[i]) ok++;
              }
              const meanPred = sumPred / X.length;
              const acc = ok / X.length;
              status.meanPred = meanPred;
              status.accuracy = acc;
              status.samples = X.length;
            } catch (err) {
              if (isDev()) console.debug("mlp: runtime status error", err);
            }
            }
          }
          (window as unknown as { mlvStatus?: Record<string, unknown> }).mlvStatus = status;
        } catch (err) {
          if (isDev()) console.debug("mlp: status publish error", err);
        }

        if (!gridG)
          gridG = p.createGraphics
            ? (p.createGraphics(p.width, p.height) as P5Graphics)
            : null;
        // ensure current step reflects adaptive or override
        gridStep = overrideStep ?? adaptiveStep;
        if (p.frameCount % gridFreq === 0 || gridDirty) {
          gridDirty = false;
          gridG.clear();

          // Darker decision region palette (aligned with Perceptron changes)
          // Dark mode: deeper, lower brightness with moderate opacity
          // Light mode: move from very pale high-opacity to richer mid-tone with lower opacity
          const blueRGBA = isDark
            ? "rgba(50,110,210,0.38)"
            : "rgba(120,170,230,0.55)";
          const redRGBA = isDark
            ? "rgba(235,70,70,0.38)"
            : "rgba(230,120,120,0.55)";

          gridG.noStroke();
          // compute grid cell counts for smoothing buffer
          const cols = Math.ceil(p.width / gridStep);
          const rows = Math.ceil(p.height / gridStep);
          if (!prevPreds || prevCols !== cols || prevRows !== rows) {
            prevPreds = new Float32Array(cols * rows);
            prevCols = cols;
            prevRows = rows;
            // initialize to neutral probability
            for (let i = 0; i < prevPreds.length; i++) prevPreds[i] = 0.5;
          }

          for (let gx = 0, cx = 0; gx < p.width; gx += gridStep, cx++) {
            for (let gy = 0, cy = 0; gy < p.height; gy += gridStep, cy++) {
              const nx = p.map(gx + 0.5 * gridStep, 0, p.width, -1, 1);
              const ny = p.map(gy + 0.5 * gridStep, p.height, 0, -1, 1);
              let prob = 0.5;
              try {
                if (model) {
                  const pval = model.forward([nx, ny]);
                  prob = Number.isFinite(pval) ? pval : 0.5;
                }
              } catch {
                prob = 0.5;
              }
              const idx = cx + cy * cols;
              const prev = prevPreds[idx] ?? 0.5;
              const blended = prev * (1 - PRED_SMOOTH) + prob * PRED_SMOOTH;
              prevPreds[idx] = blended;
              const isA = blended >= 0.5;
              gridG.fill(isA ? redRGBA : blueRGBA);
              const rectWidth = Math.min(gridStep, p.width - gx);
              const rectHeight = Math.min(gridStep, p.height - gy);
              gridG.rect(gx, gy, rectWidth, rectHeight);
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
        if (
          p.mouseX < 0 ||
          p.mouseY < 0 ||
          p.mouseX > p.width ||
          p.mouseY > p.height
        )
          return;
        const mx = p.map(p.mouseX, 0, p.width, -1, 1);
        const my = p.map(p.mouseY, p.height, 0, -1, 1);
        // on touch devices use selected touchClass; p.touches exists when touching
        let label = "A";
        try {
          const isTouch = p.touches && p.touches.length > 0;
          if (isTouch) {
            label = p._mlpControls?.touchClass ?? "A";
          } else {
            label = p.mouseButton === p.LEFT ? "A" : "B";
          }
        } catch {
          // Ignore touch detection errors
          label = p.mouseButton === p.LEFT ? "A" : "B";
        }
        points.push({ x: mx, y: my, label });
        X.push([mx, my]);
        y.push(label === "A" ? 1 : 0);
        gridDirty = true;
        if (onDatasetChange)
          onDatasetChange((d: Point[] = []) => [...d, { x: mx, y: my, label }]);
        // No automatic retrain on add (reverted)
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
        // hide finished overlay at the start of a new training session
        try {
          if (finishedRef.current) finishedRef.current.style.display = "none";
        } catch (err) {
          if (isDev()) console.debug("mlp: finishedRef hide error", err);
        }
        if (!model) resetModel();
        if (!model || !X.length) return;
        // Set up training parameters for real-time animation
        // ensure a sane epochs value
        const requested = Number(opts.epochs ?? NaN);
        const defaultEpochs = Math.max(10, Math.floor(30 / Math.max(1, speedScale)));
        trainingEpochsRemaining = Number.isFinite(requested) && requested > 0 ? Math.max(1, Math.floor(requested)) : defaultEpochs;
        trainingLR = opts.lr ?? 0.3;
        samplesInCurrentEpoch = 0;
        isTraining = true;
        // ensure the draw loop is running so training proceeds immediately
        try {
          if (typeof p.loop === 'function') p.loop();
        } catch (err) {
          if (isDev()) console.debug('mlp: p.loop error', err);
        }
        if (isDev()) {
          try { console.debug('mlp: trainMLP called', { epochs: trainingEpochsRemaining, lr: trainingLR, samples: X.length }); } catch (err) { console.debug('mlp: trainMLP debug log failed', err); }
        }
        try {
          startedAtMs =
            performance && performance.now ? performance.now() : Date.now();
        } catch (err) {
          if (isDev()) console.debug('mlp: performance.now probe error', err);
          startedAtMs = Date.now();
        }
        try {
          setLastDurationSec(null);
        } catch (err) {
          if (isDev()) console.debug("mlp: setLastDurationSec error", err);
        }
        gridDirty = true;
      };

      // Full-fit: run a blocking, deterministic fit using the classifier's fit() method.
      // This runs synchronously and may block the UI briefly for larger epoch counts,
      // but converges reliably to a low loss / perfect training accuracy for small datasets.
      (p as unknown as { fullFit?: (opts?: { epochs?: number; lr?: number; batchSize?: number; maxAttempts?: number }) => void }).fullFit = (opts: { epochs?: number; lr?: number; batchSize?: number; maxAttempts?: number } = {}) => {
        if (!model || !X.length) return;
        // prevent mixing animated training with full-fit
        const wasTraining = isTraining;
        isTraining = false;
        let startMs = Date.now();
        const epochs = opts.epochs ?? Math.max(100, controlsRef.current?.epochs ?? 200);
        const lrVal = opts.lr ?? (controlsRef.current?.lr ?? 0.3);
        // default to full-batch deterministic updates for reliable convergence
        const batchSizeVal = opts.batchSize ?? X.length;
        const maxAttempts = Math.max(1, opts.maxAttempts ?? 3);
        try {
          let finalAccuracy = 0;
          let lastLoss: number | undefined = undefined;
          for (let attempt = 1; attempt <= maxAttempts; attempt++) {
            // perform a blocking fit; model.fit will call trainSample synchronously
            const losses = model.fit(X, y, { epochs, lr: lrVal, batchSize: batchSizeVal, shuffle: false, onEpoch: (ep, L) => {
              try {
                p._lastLoss = L;
                if (ep % 10 === 0) gridDirty = true;
              } catch (err) {
                if (isDev()) console.debug('mlp: onEpoch hook error', err);
              }
            } });
            try { lastLoss = Array.isArray(losses) && losses.length ? losses[losses.length - 1] : lastLoss; } catch (err) { if (isDev()) console.debug('mlp: fullFit lastLoss calc error', err); }
            try { finalAccuracy = model.accuracy(X, y); } catch (err) { if (isDev()) console.debug('mlp: fullFit accuracy calc error', err); }
            // stop early if perfect accuracy reached
            if (finalAccuracy >= 1.0) break;
            // otherwise continue another deterministic pass (no shuffle)
            if (isDev()) {
              try { console.debug('mlp: fullFit attempt', attempt, { lastLoss, finalAccuracy }); } catch (err) { if (isDev()) console.debug('mlp: fullFit attempt log failed', err); }
            }
          }
          const elapsed = (Date.now() - startMs) / 1000;
          try {
            window.dispatchEvent(
              new CustomEvent('mlv:demo-finished', { detail: { classifier: 'Neural Network (MLP)', reason: 'train-complete', accuracy: finalAccuracy, elapsedSec: elapsed } })
            );
          } catch (err) {
            if (isDev()) console.debug('mlp: prevPreds reset error', err);
          }
          try {
            if (finishedRef.current) {
              finishedRef.current.innerText = typeof p._lastLoss === 'number'
                ? `Training finished — Loss: ${p._lastLoss.toFixed(4)} • Time: ${elapsed.toFixed(2)}s`
                : `Training finished • Time: ${elapsed.toFixed(2)}s`;
              finishedRef.current.style.display = 'block';
            }
            } catch (err) {
              if (isDev()) console.debug('mlp: control init error', err);
            }
          gridDirty = true;
        } catch (err) {
          if (isDev()) console.debug('mlp: fullFit error', err);
        } finally {
          isTraining = wasTraining;
        }
      };

      p.resetDemo = () => {
        // Clear background grid and loss immediately
        try {
          if (gridG && typeof gridG.clear === "function") gridG.clear();
        } catch (err) {
          if (isDev()) console.debug("mlp: grid clear error", err);
        }
        try {
          p._lastLoss = undefined;
        } catch (err) {
          if (isDev()) console.debug("mlp: lastLoss reset error", err);
        }
        isTraining = false;
        trainingEpochsRemaining = 0;
        gridDirty = true;
        resetDemo();
        if (onDatasetChange) onDatasetChange([]);
        try {
          if (finishedRef.current) finishedRef.current.style.display = "none";
        } catch (err) {
          if (isDev()) console.debug("mlp: finishedRef hide error", err);
        }
        try {
          setLastDurationSec(null);
        } catch (err) {
          if (isDev()) console.debug("mlp: setLastDurationSec error", err);
        }
      };
    };

    let mounted = true;
    const localRoot = sketchRef.current;
    setLoading(true);
    (async () => {
      try {
        const mod = await loadP5();
        if (!mounted) return;
    // `mod` can be either the p5 constructor or a module with a default export.
    // Use an unknown-based cast for the runtime import to avoid `any`.
    const P5 = (mod && (mod as unknown as { default?: unknown }).default) || mod;
        if (p5Ref.current) {
          p5Ref.current.remove();
          p5Ref.current = null;
        }
        const root = localRoot;
        // P5 may be a module default or the constructor itself. Narrow to a callable constructor
        type P5Constructor = new (sketch: (p: P5Instance) => void, element?: Element) => unknown;
        const P5Ctor = (P5 as unknown) as P5Constructor;
        p5Ref.current = new P5Ctor(sketch as unknown as (p: P5Instance) => void, root as Element) as unknown as P5Instance;
        setLoading(false);
        setLoadError(null);
      } catch (err) {
        if (isDev()) console.debug("mlp: dynamic import p5 failed", err);
        setLoadError(typeof err === 'string' ? err : String(err));
        setLoading(false);
      }
    })();

    return () => {
      mounted = false;
      const root = localRoot;
      if (root) {
        const canvas = root.querySelector("canvas");
        if (canvas) {
          const canvasExt = canvas as unknown as {
            _mlpPointerHandler?: (e: Event) => void;
            _mlpContextHandler?: (e: Event) => void;
          };
          const ph = canvasExt._mlpPointerHandler;
          const ch = canvasExt._mlpContextHandler;
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

  // keep controlsRef current with UI state without forcing the sketch to rebuild
  // We intentionally sync state into a ref to avoid rebuilding the heavy p5 sketch when controls change.
  useEffect(() => {
    const prevSet =
      controlsRef.current?.setGridStep ??
      ((v: number) => {
        void v;
      });
    controlsRef.current = {
      optimizer,
      activation,
      lr,
      batchSize,
      epochs,
      hiddenUnits,
      touchClass,
      setGridStep: prevSet,
    };
  }, [
    optimizer,
    activation,
    lr,
    batchSize,
    epochs,
    hiddenUnits,
    touchClass,
    gridStepState,
  ]);

  // Pause p5 draw loop when tab is hidden to reduce CPU usage
  useEffect(() => {
    const onVis = () => {
      try {
        const inst = p5Ref.current as P5Instance | null;
        if (
          !inst ||
          typeof inst.noLoop !== "function" ||
          typeof inst.loop !== "function"
        )
          return;
        if ((document as Document & { hidden?: boolean }).hidden) inst.noLoop();
        else inst.loop();
      } catch (err) {
        if (isDev()) console.debug("mlp: visibility handler error", err);
      }
    };
    document.addEventListener("visibilitychange", onVis);
    onVis();
    return () => document.removeEventListener("visibilitychange", onVis);
  }, []);

  useEffect(() => {
    if (p5Ref.current && typeof p5Ref.current.updateDataset === "function") {
      p5Ref.current.updateDataset(externalDataset || []);
    }
  }, [externalDataset]);

  // poll p5 instance for lastLoss set by training callback and update React state
  useEffect(() => {
    const int = setInterval(() => {
      if (p5Ref.current) {
        const L = p5Ref.current._lastLoss;
        if (typeof L === "number") setLoss(L);
        // show debug status in dev mode
        try {
          // Touch the p5 instance status field in dev mode (no-op) without creating a local unused variable
          void ((p5Ref.current as unknown as { _mlpStatus?: Record<string, unknown> })?._mlpStatus);
        } catch (err) {
          if (isDev()) console.debug('mlp: dynamic import p5 failed', err);
        }
      }
    }, 200);
    return () => clearInterval(int);
  }, []);

  // poll window.mlvStatus for runtime diagnostics published by the sketch
  useEffect(() => {
    if (!isDev()) return;
    const t = setInterval(() => {
        try {
          const st = (window as unknown as { mlvStatus?: Record<string, unknown> }).mlvStatus;
          if (st) {
            if (typeof st.meanPred === 'number') setMlpMeanPred(st.meanPred as number);
            if (typeof st.accuracy === 'number') setMlpAcc(st.accuracy as number);
          }
        } catch (err) {
          if (isDev()) console.debug('mlp: window.mlvStatus poll error', err);
        }
    }, 300);
    return () => clearInterval(t);
  }, []);

  // auto-fallback if model collapses to a constant prediction (dev only)
  const triedFallbackRef = useRef(false);
  useEffect(() => {
    if (!isDev()) return;
    if (triedFallbackRef.current) return;
    if (mlpMeanPred == null || mlpAcc == null) return;
    // detect collapse: meanPred very close to 0 or 1 and low accuracy
    if ((mlpMeanPred < 0.05 || mlpMeanPred > 0.95) && mlpAcc < 0.7) {
      triedFallbackRef.current = true;
      // switch to SGD with a standard lr and retrain
      setOptimizer('sgd');
      setLr(0.3);
      // trigger retrain using p5 instance
      setTimeout(() => {
        const inst = p5Ref.current as P5Instance | null;
        try {
          if (inst && typeof inst.resetDemo === 'function') {
            inst.resetDemo();
            if (typeof inst.trainMLP === 'function') inst.trainMLP({ epochs: 50 });
          }
        } catch (err) {
          if (isDev()) console.debug('mlp: auto-fallback error', err);
        }
      }, 200);
    }
  }, [mlpMeanPred, mlpAcc]);

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
    const inst = p5Ref.current;
    if (inst && typeof inst.trainMLP === "function") {
      const t = setTimeout(
        () => {
          const ctrlEpochs = controlsRef.current?.epochs ?? 100;
          inst.trainMLP({ epochs: Math.max(8, Math.floor(Number(ctrlEpochs) / Math.max(1, speedScale))) });
        },
        50
      );
      return () => clearTimeout(t);
    }
    return;
  }, [resetToken, speedScale]);

  const containerStyle: React.CSSProperties = compact
    ? {
        position: "relative",
        width: "100%",
        height: "100%",
        overflow: "hidden",
        marginLeft: 0,
      }
    : {
        position: "relative",
        width: "calc(100% - 280px)",
        height: "100%",
        overflow: "hidden",
        // shift a bit more to the left so the canvas is visually centered when the
        // floating control panel is present on the right
        marginLeft: "-200px",
      };

  return (
    <div ref={sketchRef} style={containerStyle}>
      {loading && (
        <div
          style={{
            position: "absolute",
            inset: 0,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            background: "rgba(255,255,255,0.6)",
            zIndex: 2000,
          }}
        >
          <div
            style={{
              padding: 12,
              borderRadius: 8,
              background: "rgba(0,0,0,0.6)",
              color: "#fff",
              fontWeight: 600,
            }}
          >
            Loading demo…
          </div>
        </div>
      )}
      {loadError && (
        <div
          style={{
            position: "absolute",
            inset: 0,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            background: "rgba(0,0,0,0.45)",
            zIndex: 2100,
            color: "#fff",
            padding: 20,
          }}
        >
          <div
            style={{
              maxWidth: 680,
              background: "rgba(0,0,0,0.6)",
              padding: 18,
              borderRadius: 8,
              boxShadow: "0 8px 32px rgba(0,0,0,0.6)",
            }}
          >
            <div style={{ fontWeight: 700, marginBottom: 8 }}>Demo failed to load</div>
            <div style={{ marginBottom: 12 }}>{String(loadError)}</div>
            <div style={{ fontSize: 13, opacity: 0.9 }}>
              Try refreshing the page or check the browser console for details.
            </div>
          </div>
          {isDev() && (
            <div
              style={{
                position: 'absolute',
                right: 280,
                top: 12,
                zIndex: 60,
                padding: '8px 10px',
                background: 'rgba(0,0,0,0.6)',
                color: '#fff',
                borderRadius: 8,
                fontSize: 12,
                pointerEvents: 'none'
              }}
            >
              <div style={{opacity:0.9}}>MLP debug</div>
              <div style={{fontFamily:'monospace', fontSize:12}}>
                MeanPred: {mlpMeanPred == null ? '—' : mlpMeanPred.toFixed(3)}
              </div>
              <div style={{fontFamily:'monospace', fontSize:12}}>
                Acc: {mlpAcc == null ? '—' : (mlpAcc*100).toFixed(1) + '%'}
              </div>
            </div>
          )}
        </div>
      )}
      <div
        style={{
          position: "absolute",
          left: 12,
          top: 12,
          zIndex: 20,
          display: "flex",
          gap: 8,
        }}
      >
        <button
          onClick={() => {
              setLoss(null);
              // reset runtime diagnostics shown in the UI
                try {
                  setMlpAcc(null);
                  setMlpMeanPred(null);
                  // also clear the published global status if present (narrow cast)
                  try {
                    (window as unknown as { mlvStatus?: unknown }).mlvStatus = undefined;
                  } catch (err) {
                    if (isDev()) console.debug('mlp: clear mlvStatus error', err);
                  }
                } catch (err) {
                  if (isDev()) console.debug('mlp: reset diagnostics error', err);
                }
              if (
                p5Ref.current &&
                typeof p5Ref.current.resetDemo === "function"
              ) {
                try {
                  // suppress status publishing for 400ms so overlays stay cleared after reset
                  const inst = p5Ref.current as unknown as { suppressStatus?: (ms: number) => void } | null;
                  if (inst && typeof inst.suppressStatus === 'function') inst.suppressStatus(400);
                } catch (err) {
                  if (isDev()) console.debug('mlp: suppressStatus error', err);
                }
              p5Ref.current.resetDemo();
              }
          }}
          style={{
            padding: "8px 16px",
            fontSize: 12,
            fontWeight: 600,
            background: prefersDark
              ? "rgba(45,55,72,0.95)"
              : "rgba(255,255,255,0.95)",
            color: panelText,
            border: panelBorder,
            borderRadius: 8,
            cursor: "pointer",
            backdropFilter: "blur(10px)",
            boxShadow: "0 2px 8px rgba(0,0,0,0.15)",
            transition: "all 0.2s ease",
          }}
          onMouseOver={(e) => {
            e.currentTarget.style.transform = "translateY(-1px)";
            e.currentTarget.style.boxShadow = "0 4px 12px rgba(0,0,0,0.2)";
          }}
          onMouseOut={(e) => {
            e.currentTarget.style.transform = "translateY(0)";
            e.currentTarget.style.boxShadow = "0 2px 8px rgba(0,0,0,0.15)";
          }}
        >
          🔄 Reset
        </button>
        <button
            onClick={() => {
              // run a blocking, full-fit on the current model/dataset
              try {
                const ctrlEpochs = controlsRef.current?.epochs ?? 200;
                const inst = p5Ref.current as unknown as { fullFit?: (opts?: { epochs?: number; lr?: number; batchSize?: number }) => void } | null;
                if (inst && typeof inst.fullFit === 'function') {
                  inst.fullFit({ epochs: Math.max(50, Number(ctrlEpochs)), lr: controlsRef.current?.lr, batchSize: controlsRef.current?.batchSize ?? undefined });
                }
              } catch (err) {
                try {
                  if (isDev()) console.debug('mlp: full-fit click error', err);
                } catch (err2) {
                  if (isDev()) console.debug('mlp: full-fit click debug error', err2);
                }
              }
            }}
          style={{
            padding: "8px 16px",
            fontSize: 12,
            fontWeight: 600,
            background: prefersDark ? "rgba(0,0,0,0.6)" : "rgba(255,255,255,0.95)",
            color: prefersDark ? '#fff' : '#1a202c',
            border: panelBorder,
            borderRadius: 8,
            cursor: "pointer",
            boxShadow: "0 2px 8px rgba(0,0,0,0.12)",
            transition: "all 0.15s ease",
          }}
        >
          ⚡ Full-fit
        </button>
        <button
            onClick={() => {
              const ctrlEpochs = controlsRef.current?.epochs ?? 100;
              if (p5Ref.current && typeof p5Ref.current.trainMLP === "function") {
                p5Ref.current.trainMLP({ epochs: Number(ctrlEpochs) });
              }
            }}
          style={{
            padding: "8px 16px",
            fontSize: 12,
            fontWeight: 600,
            background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
            color: "#fff",
            border: "none",
            borderRadius: 8,
            cursor: "pointer",
            boxShadow: "0 2px 8px rgba(102, 126, 234, 0.4)",
            transition: "all 0.2s ease",
          }}
          onMouseOver={(e) => {
            e.currentTarget.style.transform = "translateY(-1px)";
            e.currentTarget.style.boxShadow =
              "0 4px 12px rgba(102, 126, 234, 0.5)";
          }}
          onMouseOut={(e) => {
            e.currentTarget.style.transform = "translateY(0)";
            e.currentTarget.style.boxShadow =
              "0 2px 8px rgba(102, 126, 234, 0.4)";
          }}
        >
          ▶️ Train
        </button>
      </div>
      {showInstructions && (
        <div
          style={{
            position: "absolute",
            left: 12,
            top: 60,
            zIndex: 20,
            maxWidth: 280,
            background: prefersDark
              ? "rgba(45,55,72,0.95)"
              : "rgba(255,255,255,0.95)",
            backdropFilter: "blur(10px)",
            borderRadius: 10,
            padding: "10px 12px",
            border: panelBorder,
            boxShadow: controlShadow,
          }}
        >
          <div style={{ fontSize: 12, color: panelText, lineHeight: 1.5 }}>
            <div
              style={{
                fontWeight: 600,
                marginBottom: 4,
                fontSize: 11,
                opacity: 0.7,
                textTransform: "uppercase",
                letterSpacing: "0.5px",
              }}
            >
              🧠 Neural Network
            </div>
            <div style={{ fontSize: 11, opacity: 0.9 }}>
              <span style={{ fontWeight: 600 }}>Left click:</span> Add Red (A)
              <br />
              <span style={{ fontWeight: 600 }}>Right click:</span> Add Blue (B)
            </div>
          </div>
        </div>
      )}
      {/* control panel */}
      <div
        ref={panelRef}
        style={{
          position: "fixed",
          right: 16,
          top: 86,
          zIndex: 1150,
          touchAction: "none",
          cursor: "default",
          maxHeight: "calc(100vh - 180px)",
          overflowY: "auto",
          display: compact ? "none" : undefined,
        }}
      >
        <div
          style={{
            padding: "12px",
            background: panelBg,
            borderRadius: "12px",
            width: 240,
            boxSizing: "border-box",
            color: panelText,
            border: panelBorder,
            boxShadow: controlShadow,
          }}
        >
          <div
            style={{
              marginBottom: 10,
              paddingBottom: 8,
              borderBottom: `1px solid ${
                prefersDark ? "rgba(255,255,255,0.1)" : "rgba(0,0,0,0.1)"
              }`,
              fontSize: 13,
              fontWeight: 600,
              letterSpacing: "0.3px",
            }}
          >
            ⚙️ Model Controls
          </div>

          {/* Legacy / Stable toggle */}
          <div style={{ marginBottom: 10, display: 'flex', alignItems: 'center', gap: 8 }}>
            <input
              id="legacyStable"
              type="checkbox"
              checked={legacyStable}
              onChange={(e) => {
                const v = e.target.checked;
                setLegacyStable(v);
                if (v) {
                  // apply proven safe defaults
                  setOptimizer('sgd');
                  setLr(0.3);
                  setHiddenUnits(16);
                  setActivation('sigmoid');
                  setEpochs(20);
                  setBatchSize(8);
                  // ensure the running sketch picks up the new controls and resets
                  const inst = p5Ref.current as P5Instance | null;
                  try {
                    if (inst && typeof inst.resetDemo === 'function') inst.resetDemo();
                  } catch (err) {
                    if (isDev()) console.debug('mlp: legacy toggle reset error', err);
                  }
                }
              }}
            />
            <label htmlFor="legacyStable" style={{ fontSize: 12, opacity: 0.9 }}>
              Legacy / Stable MLP (safe defaults)
            </label>
          </div>

          {/* Architecture Section */}
          <div style={{ marginBottom: 12 }}>
            <div
              style={{
                fontSize: 11,
                fontWeight: 600,
                marginBottom: 6,
                opacity: 0.7,
                textTransform: "uppercase",
                letterSpacing: "0.5px",
              }}
            >
              Architecture
            </div>
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "1fr 1fr",
                gap: 6,
              }}
            >
              <div>
                <label
                  style={{
                    fontSize: 11,
                    color: panelText,
                    opacity: 0.8,
                    display: "block",
                    marginBottom: 2,
                  }}
                >
                  Activation
                </label>
                <select
                  value={activation}
                  onChange={(e) => {
                    setActivation(e.target.value);
                    if (p5Ref.current) {
                      const inst = p5Ref.current as P5Instance;
                      if (inst._mlpControls)
                        inst._mlpControls.activation = e.target.value;
                      if (typeof inst.resetDemo === "function") {
                        inst.resetDemo();
                      }
                    }
                  }}
                  style={{
                    width: "100%",
                    padding: "4px 6px",
                    fontSize: 11,
                    background: inputBg,
                    color: inputColor,
                    borderRadius: 6,
                    border: panelBorder,
                  }}
                >
                  <option value="sigmoid">Sigmoid</option>
                  <option value="tanh">Tanh</option>
                  <option value="relu">ReLU</option>
                  <option value="leaky_relu">Leaky ReLU</option>
                </select>
              </div>
              <div>
                <label
                  style={{
                    fontSize: 11,
                    color: panelText,
                    opacity: 0.8,
                    display: "block",
                    marginBottom: 2,
                  }}
                >
                  Hidden Units
                </label>
                <input
                  type="number"
                  value={hiddenUnits}
                  onChange={(e) => {
                    const v = Number(e.target.value);
                    setHiddenUnits(v);
                    if (p5Ref.current) {
                      const inst = p5Ref.current as P5Instance;
                      if (inst._mlpControls) inst._mlpControls.hiddenUnits = v;
                    }
                  }}
                  style={{
                    width: "100%",
                    padding: "4px 6px",
                    fontSize: 11,
                    background: inputBg,
                    color: inputColor,
                    borderRadius: 6,
                    border: panelBorder,
                    boxSizing: "border-box",
                  }}
                />
              </div>
            </div>
          </div>

          {/* Training Section */}
          <div style={{ marginBottom: 12 }}>
            <div
              style={{
                fontSize: 11,
                fontWeight: 600,
                marginBottom: 6,
                opacity: 0.7,
                textTransform: "uppercase",
                letterSpacing: "0.5px",
              }}
            >
              Training
            </div>
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "1fr 1fr",
                gap: 6,
                marginBottom: 6,
              }}
            >
              <div>
                <label
                  style={{
                    fontSize: 11,
                    color: panelText,
                    opacity: 0.8,
                    display: "block",
                    marginBottom: 2,
                  }}
                >
                  Optimizer
                </label>
                <select
                  value={optimizer}
                  onChange={(e) => {
                    setOptimizer(e.target.value);
                    if (p5Ref.current) {
                      const inst = p5Ref.current as P5Instance;
                      if (inst._mlpControls)
                        inst._mlpControls.optimizer = e.target.value;
                    }
                  }}
                  style={{
                    width: "100%",
                    padding: "4px 6px",
                    fontSize: 11,
                    background: inputBg,
                    color: inputColor,
                    borderRadius: 6,
                    border: panelBorder,
                  }}
                >
                  <option value="sgd">SGD</option>
                  <option value="momentum">Momentum</option>
                  <option value="adam">Adam</option>
                </select>
              </div>
              <div>
                <label
                  style={{
                    fontSize: 11,
                    color: panelText,
                    opacity: 0.8,
                    display: "block",
                    marginBottom: 2,
                  }}
                >
                  Learning Rate
                </label>
                <input
                  type="number"
                  step="0.01"
                  value={lr}
                  onChange={(e) => {
                    const v = Number(e.target.value);
                    setLr(v);
                    if (p5Ref.current) {
                      const inst = p5Ref.current as P5Instance;
                      if (inst._mlpControls) inst._mlpControls.lr = v;
                    }
                  }}
                  style={{
                    width: "100%",
                    padding: "4px 6px",
                    fontSize: 11,
                    background: inputBg,
                    color: inputColor,
                    borderRadius: 6,
                    border: panelBorder,
                    boxSizing: "border-box",
                  }}
                />
              </div>
            </div>
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "1fr 1fr",
                gap: 6,
              }}
            >
              <div>
                <label
                  style={{
                    fontSize: 11,
                    color: panelText,
                    opacity: 0.8,
                    display: "block",
                    marginBottom: 2,
                  }}
                >
                  Batch Size
                </label>
                <input
                  type="number"
                  value={batchSize}
                  onChange={(e) => {
                    const v = Number(e.target.value);
                    setBatchSize(v);
                    if (p5Ref.current) {
                      const inst = p5Ref.current as P5Instance;
                      if (inst._mlpControls) inst._mlpControls.batchSize = v;
                    }
                  }}
                  style={{
                    width: "100%",
                    padding: "4px 6px",
                    fontSize: 11,
                    background: inputBg,
                    color: inputColor,
                    borderRadius: 6,
                    border: panelBorder,
                    boxSizing: "border-box",
                  }}
                />
              </div>
              <div>
                <label
                  style={{
                    fontSize: 11,
                    color: panelText,
                    opacity: 0.8,
                    display: "block",
                    marginBottom: 2,
                  }}
                >
                  Epochs
                </label>
                <input
                  type="number"
                  value={epochs}
                  onChange={(e) => {
                    const v = Number(e.target.value);
                    setEpochs(v);
                    if (p5Ref.current) {
                      const inst = p5Ref.current as P5Instance;
                      if (inst._mlpControls) inst._mlpControls.epochs = v;
                    }
                  }}
                  style={{
                    width: "100%",
                    padding: "4px 6px",
                    fontSize: 11,
                    background: inputBg,
                    color: inputColor,
                    borderRadius: 6,
                    border: panelBorder,
                    boxSizing: "border-box",
                  }}
                />
              </div>
            </div>
          </div>

          {/* Visualization Section */}
          <div style={{ marginBottom: 8 }}>
            <div
              style={{
                fontSize: 11,
                fontWeight: 600,
                marginBottom: 6,
                opacity: 0.7,
                textTransform: "uppercase",
                letterSpacing: "0.5px",
              }}
            >
              Visualization
            </div>
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "1fr 1fr",
                gap: 6,
              }}
            >
              <div>
                <label
                  style={{
                    fontSize: 11,
                    color: panelText,
                    opacity: 0.8,
                    display: "block",
                    marginBottom: 2,
                  }}
                >
                  Grid Size
                </label>
                <input
                  type="number"
                  value={gridStepState}
                  onChange={(e) => {
                    const v = Number(e.target.value);
                    setGridStepState(v);
                    if (p5Ref.current) {
                      const inst = p5Ref.current as P5Instance;
                      inst._mlpControls?.setGridStep?.(v);
                    }
                  }}
                  style={{
                    width: "100%",
                    padding: "4px 6px",
                    fontSize: 11,
                    background: inputBg,
                    color: inputColor,
                    borderRadius: 6,
                    border: panelBorder,
                    boxSizing: "border-box",
                  }}
                />
              </div>
              <div>
                <label
                  style={{
                    fontSize: 11,
                    color: panelText,
                    opacity: 0.8,
                    display: "block",
                    marginBottom: 2,
                  }}
                >
                  Touch Adds
                </label>
                <select
                  value={touchClass}
                  onChange={(e) => {
                    const v = e.target.value as "A" | "B";
                    setTouchClass(v);
                    if (p5Ref.current) {
                      const inst = p5Ref.current as P5Instance;
                      if (inst._mlpControls) inst._mlpControls.touchClass = v;
                    }
                  }}
                  style={{
                    width: "100%",
                    padding: "4px 6px",
                    fontSize: 11,
                    background: inputBg,
                    color: inputColor,
                    borderRadius: 6,
                    border: panelBorder,
                  }}
                >
                  <option value="A">🔴 Red</option>
                  <option value="B">🔵 Blue</option>
                </select>
              </div>
            </div>
          </div>
        </div>
      </div>
      {/* live loss overlay */}
      <div
        style={{
          position: "absolute",
          left: 12,
          bottom: 12,
          zIndex: 30,
          padding: "8px 12px",
          background: prefersDark
            ? "rgba(45,55,72,0.95)"
            : "rgba(255,255,255,0.95)",
          color: panelText,
          borderRadius: 8,
          fontSize: 11,
          fontWeight: 500,
          border: panelBorder,
          backdropFilter: "blur(10px)",
          boxShadow: "0 2px 8px rgba(0,0,0,0.15)",
        }}
      >
        <span style={{ opacity: 0.7 }}>Loss:</span>{" "}
        <span style={{ fontWeight: 600, fontFamily: "monospace" }}>
          {loss == null ? "—" : loss.toFixed(4)}
        </span>
      </div>
      {/* finished overlay */}
      <div
        ref={finishedRef}
        style={{
          position: "absolute",
          top: 10,
          left: "50%",
          transform: "translateX(-50%)",
          background: "rgba(0,0,0,0.6)",
          color: "#fff",
          padding: "8px 12px",
          borderRadius: 8,
          fontSize: 12,
          letterSpacing: 0.2,
          display: "none",
          zIndex: 35,
          boxShadow: "0 4px 12px rgba(0,0,0,0.25)",
        }}
      >
        Training finished
      </div>
    </div>
  );
};

export default MlpDemo;
