import React, { useEffect, useRef, useState } from "react";
// The MLP sketch exposes a number of control getters/setters on the p5 instance
// which intentionally read from refs to avoid rebuilding the heavy sketch when UI controls change.
import MLPClassifier, { ActivationFunction } from "../utils/mlp";
import type { P5Instance } from "../types/p5";
import type { P5Graphics } from "../types";

type Point = { x: number; y: number; label?: string };

type MlpDemoProps = {
  showInstructions?: boolean;
  speedScale?: number;
  dataset?: Point[];
  onDatasetChange?: (updater: ((d: Point[]) => Point[]) | Point[]) => void;
  resetToken?: unknown;
};

const MlpDemo: React.FC<MlpDemoProps> = ({
  showInstructions = true,
  speedScale = 1,
  dataset: externalDataset,
  onDatasetChange,
  resetToken,
}) => {
  const sketchRef = useRef<HTMLDivElement | null>(null);
  const p5Ref = useRef<P5Instance | null>(null);
  const [loading, setLoading] = useState(true);
  const [loss, setLoss] = useState<number | null>(null);
  const finishedRef = useRef<HTMLDivElement | null>(null);
  // timing overlay for training duration
  const [_lastDurationSec, setLastDurationSec] = useState<number | null>(null);
  // control panel state
  const [optimizer, setOptimizer] = useState<string>("sgd");
  const [activation, setActivation] = useState<string>("sigmoid");
  const [lr, setLr] = useState<number>(0.3);
  const [batchSize, setBatchSize] = useState<number>(8);
  const [epochs, setEpochs] = useState<number>(20);
  const [hiddenUnits, setHiddenUnits] = useState<number>(16);
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

      const resetModel = (hidden: number[] = []) => {
        // Read control values from controlsRef to avoid closing over React state in the sketch.
        const ctrl = controlsRef.current;
        const lrVal = ctrl?.lr ?? 0.3;
        const opt = ctrl?.optimizer ?? "sgd";
        const act = (ctrl?.activation ?? "sigmoid") as ActivationFunction;
        const hiddenUnitsVal = ctrl?.hiddenUnits ?? 16;
        const h = hidden.length ? hidden : [hiddenUnitsVal];
        model = new MLPClassifier(2, h, {
          lr: lrVal,
          optimizer: opt,
          activation: act,
        });
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
          };
          p._mlpControls = {
            // Use ref-backed values; fall back to stable literals so the sketch-creation effect
            // doesn't close over React state variables (avoids rebuilds and satisfies exhaustive-deps).
            get optimizer() {
              return controlsRef.current?.optimizer ?? "sgd";
            },
            get activation() {
              return controlsRef.current?.activation ?? "sigmoid";
            },
            get lr() {
              return controlsRef.current?.lr ?? 0.3;
            },
            get batchSize() {
              return controlsRef.current?.batchSize ?? 8;
            },
            get epochs() {
              return controlsRef.current?.epochs ?? 20;
            },
            get hiddenUnits() {
              return controlsRef.current?.hiddenUnits ?? 16;
            },
            get touchClass() {
              return controlsRef.current?.touchClass ?? "A";
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
        p.background(isDark ? 30 : 255);

        // Publish status for AgentPanel (MLP has no simple closed-form equation)
        try {
          (
            window as unknown as { mlvStatus?: Record<string, unknown> }
          ).mlvStatus = {
            classifier: "mlp",
            equation: null,
            weights: undefined,
            bias: undefined,
            updatedAt: Date.now(),
          };
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

          // Match Perceptron background palette for decision regions
          // Perceptron uses:
          //  - Dark: classA "rgba(80,150,255,0.3)", classB "rgba(255,100,100,0.3)"
          //  - Light: classA "rgba(200,230,255,0.9)", classB "rgba(255,220,220,0.9)"
          const blueRGBA = isDark
            ? "rgba(80,150,255,0.3)"
            : "rgba(200,230,255,0.9)";
          const redRGBA = isDark
            ? "rgba(255,100,100,0.3)"
            : "rgba(255,220,220,0.9)";

          gridG.noStroke();
          for (let gx = 0; gx < p.width; gx += gridStep) {
            for (let gy = 0; gy < p.height; gy += gridStep) {
              const nx = p.map(gx + 0.5 * gridStep, 0, p.width, -1, 1);
              const ny = p.map(gy + 0.5 * gridStep, p.height, 0, -1, 1);
              const pred = model ? (model.forward([nx, ny]) >= 0.5 ? 1 : 0) : 0;
              // Keep class mapping consistent with perceptron visuals:
              // pred === 1 -> class A (blue), pred === 0 -> class B (red)
              gridG.fill(pred === 1 ? blueRGBA : redRGBA);
              // Ensure full coverage by extending to canvas edge if needed
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
        trainingEpochsRemaining =
          opts.epochs ?? Math.max(10, Math.floor(30 / Math.max(1, speedScale)));
        trainingLR = opts.lr ?? 0.3;
        samplesInCurrentEpoch = 0;
        isTraining = true;
        try {
          startedAtMs =
            performance && performance.now ? performance.now() : Date.now();
        } catch {
          startedAtMs = Date.now();
        }
        try {
          setLastDurationSec(null);
        } catch (err) {
          if (isDev()) console.debug("mlp: setLastDurationSec error", err);
        }
        setLastDurationSec(null);
        gridDirty = true;
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
        const mod = await import("p5");
        if (!mounted) return;
        // `mod` can be either the p5 constructor or a module with a default export.
        // Use a narrow eslint disable for the `any` cast here since it's a runtime import.
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const P5 = (mod && (mod as any).default) || mod;
        if (p5Ref.current) {
          p5Ref.current.remove();
          p5Ref.current = null;
        }
        const root = localRoot;
        p5Ref.current = new P5(
          sketch,
          root as Element
        ) as unknown as P5Instance;
        setLoading(false);
      } catch (err) {
        if (isDev()) console.debug("mlp: dynamic import p5 failed", err);
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
    const inst = p5Ref.current;
    if (inst && typeof inst.trainMLP === "function") {
      const t = setTimeout(
        () =>
          inst.trainMLP({
            epochs: Math.max(8, Math.floor(20 / Math.max(1, speedScale))),
          }),
        50
      );
      return () => clearTimeout(t);
    }
    return;
  }, [resetToken, speedScale]);

  return (
    <div
      ref={sketchRef}
      style={{
        position: "relative",
        width: "calc(100% - 280px)",
        height: "100%",
        overflow: "hidden",
        marginLeft: "-140px",
      }}
    >
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
            if (
              p5Ref.current &&
              typeof p5Ref.current.resetDemo === "function"
            ) {
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
          onClick={() =>
            p5Ref.current &&
            typeof p5Ref.current.trainMLP === "function" &&
            p5Ref.current.trainMLP({ epochs: 20 })
          }
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
