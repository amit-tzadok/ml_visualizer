/* eslint-disable @typescript-eslint/no-explicit-any */
import React, { useEffect, useRef, useState } from "react";
import Perceptron from "../utils/perceptron";
import PolynomialPerceptron from "../utils/polynomialClassifier";
import styles from "./PerceptronDemo.module.css";
import type { P5Instance } from "../types/p5";
import type { P5Graphics, PerceptronAdapter } from "../types";

type Point = { x: number; y: number; label?: string };

type PerceptronDemoProps = {
  classifierType?: "linear" | "poly" | string;
  dataset?: Point[];
  onDatasetChange?: (updater: ((d: Point[]) => Point[]) | Point[]) => void;
  resetToken?: unknown;
  speedScale?: number;
  showInstructions?: boolean;
};

const PerceptronDemo: React.FC<PerceptronDemoProps> = ({
  classifierType = "linear",
  dataset: externalDataset,
  onDatasetChange,
  resetToken,
  speedScale = 1,
}) => {
  const sketchRef = useRef<HTMLDivElement | null>(null);
  const p5InstanceRef = useRef<P5Instance | null>(null);
  const equationRef = useRef<HTMLDivElement | null>(null);
  const finishedRef = useRef<HTMLDivElement | null>(null);
  const [loading, setLoading] = useState(true);
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
    const fmt = (n: number) =>
      !isFinite(n) ? "0" : Math.abs(n) < 1e-6 ? 0 : Number(n.toFixed(3));

    const buildLinearEquation = (weights: number[] = [], bias = 0) => {
      if (weights.length < 2) return "linear: no weights";
      const w0 = weights[0],
        w1 = weights[1];
      if (Math.abs(w1) > 1e-8) {
        const slope = -w0 / w1;
        const intercept = -bias / w1;
        return `y = ${fmt(slope)} x ${intercept >= 0 ? "+" : "-"} ${fmt(
          Math.abs(intercept)
        )}`;
      } else if (Math.abs(w0) > 1e-8) {
        return `x = ${fmt(-bias / w0)}`;
      }
      return "degenerate boundary";
    };

    const buildPolyEquation = (weights: number[] = [], bias = 0) => {
      const terms: string[] = [];
      const names = ["x", "y", "x^2", "y^2", "xy"];
      for (let i = 0; i < Math.min(weights.length, 5); i++) {
        if (Math.abs(weights[i]) < 1e-6) continue;
        terms.push(`${fmt(weights[i])}·${names[i]}`);
      }
      return `${terms.join(" + ") || "0"} ${bias >= 0 ? "+" : "-"} ${fmt(
        Math.abs(bias)
      )} = 0`;
    };

    const sketch = (p: P5Instance) => {
      // Local typed alias for p5 instance to allow a few runtime-extended fields
      type P5LinearExt = {
        _linLast?: {
          w0: number;
          w1: number;
          b: number;
          colStep: number;
          w: number;
          h: number;
        };
        [k: string]: unknown;
      };
      const pEx = p as unknown as P5Instance & P5LinearExt;
      let points: Array<{ x: number; y: number; labelSigned?: number }> = [],
        X: number[][] = [],
        y: number[] = [];
      let gridG: P5Graphics | null = null; // offscreen graphics for decision background (poly & linear)
      // Adaptive grid resolution for performance: coarse during training, fine when paused
      let fineGridStep = 6;
      let coarseGridStep = 14;
      let gridStep = coarseGridStep;
      let gridFreq = 18; // update less often during training
      let wasPaused = false;
      // model is a typed adapter around the raw Perceptron/PolynomialPerceptron instances
      let model: import("../types").PerceptronAdapter,
        paused = false,
        speed =
          externalDataset &&
          Array.isArray(externalDataset) &&
          externalDataset.length
            ? 1
            : 0.3,
        trainingAccumulator = 0,
        epoch = 0,
        indexInEpoch = 0,
        _errorsInEpoch = 0;
      // track mistakes per epoch to detect convergence and notify once
      let mistakesThisEpoch = 0;
      let zeroEpochStreak = 0; // require consecutive zero-mistake epochs to be robust
      let notifiedDone = false;
      // When finishing, defer heavy background repaints briefly so success animation can appear immediately
      let finishedAtMs = 0;
      // Track training start time for elapsed duration
      let startedAtMs: number | null = null;
      const MAX_EPOCHS = 1000;

      const createRawModel = () =>
        (classifierType === "poly"
          ? new PolynomialPerceptron(2, 0.05, 1)
          : new Perceptron(2, 0.05, 1)) as unknown;
      const adaptModel = (m: unknown): PerceptronAdapter => {
        // treat the incoming raw object as a partial adapter to avoid unsafe any creep
        const mm = m as Partial<
          PerceptronAdapter & { raw?: any; model?: any }
        > | null;
        return {
          raw: m,
          predict: (sample: number[]) => {
            if (!mm) return 0;
            if (typeof (mm as any).predict === "function")
              return (mm as any).predict(sample);
            if (mm.raw && typeof mm.raw.predict === "function")
              return mm.raw.predict(sample);
            return 0;
          },
          predictRaw: (sample: number[]) => {
            if (!mm) return 0;
            if (typeof (mm as any).predictRaw === "function")
              return (mm as any).predictRaw(sample);
            if (mm.raw && typeof mm.raw.predictRaw === "function")
              return mm.raw.predictRaw(sample);
            return 0;
          },
          trainSample: (sample: number[], label: number) => {
            if (!mm) return;
            if (typeof (mm as any).trainSample === "function")
              return (mm as any).trainSample(sample, label);
            if (typeof (mm as any).fit === "function")
              return (mm as any).fit([sample], [label]);
          },
          get weights() {
            return (
              mm?.weights ||
              (mm?.model as any)?.weights ||
              (mm?.raw as any)?.weights ||
              (mm?.raw as any)?.model?.weights ||
              []
            );
          },
          get bias() {
            const b =
              mm?.bias ??
              (mm?.model as any)?.bias ??
              (mm?.raw as any)?.bias ??
              (mm?.raw as any)?.model?.bias;
            return typeof b === "number" ? b : 0;
          },
          misclassificationRate: (X: number[][], y: Array<number | string>) => {
            try {
              if (mm && typeof (mm as any).misclassificationRate === "function")
                return (mm as any).misclassificationRate(X, y);
              if (
                mm &&
                mm.raw &&
                typeof mm.raw.misclassificationRate === "function"
              )
                return mm.raw.misclassificationRate(X, y);
            } catch (err) {
              if (isDev())
                console.debug("perceptron: misclassificationRate error", err);
              return NaN;
            }
            return NaN;
          },
          hingeLoss: (X: number[][], y: Array<number | string>) => {
            try {
              if (mm && typeof (mm as any).hingeLoss === "function")
                return (mm as any).hingeLoss(X, y);
              if (mm && mm.raw && typeof mm.raw.hingeLoss === "function")
                return mm.raw.hingeLoss(X, y);
            } catch (err) {
              if (isDev()) console.debug("perceptron: hingeLoss error", err);
              return NaN;
            }
            return NaN;
          },
          fitHingeSGD: (
            X: number[][],
            y: Array<number | string>,
            options?: {
              epochs?: number;
              lr?: number;
              lambda?: number;
              shuffle?: boolean;
            }
          ) => {
            if (!mm) return;
            if (typeof (mm as any).fitHingeSGD === "function")
              return (mm as any).fitHingeSGD(X, y, options);
            if (mm.raw && typeof mm.raw.fitHingeSGD === "function")
              return mm.raw.fitHingeSGD(X, y, options);
          },
        };
      };
      const getCurrentEquation = () => {
        try {
          const w = (model && model.weights) || [];
          const b = (model && model.bias) || 0;
          if (classifierType === "linear")
            return w.length >= 2
              ? buildLinearEquation(w, b)
              : "linear: insufficient weights";
          return buildPolyEquation(w, b);
        } catch (err) {
          if (isDev())
            console.debug("perceptron: getCurrentEquation error", err);
          return "";
        }
      };
      // Cached best result for Maximize Margin to keep results stable per dataset
      let lastMaximizeSig: string | null = null;
      let lastBestModel: unknown = null;

      const reset = () => {
        model = adaptModel(createRawModel());
        points = [];
        X = [];
        y = [];
        epoch = indexInEpoch = _errorsInEpoch = 0;
        zeroEpochStreak = 0;
        mistakesThisEpoch = 0;
        notifiedDone = false;
        lastMaximizeSig = null;
        lastBestModel = null;
        paused = false;
        trainingAccumulator = 0;
        wasPaused = false;
        try {
          if (gridG) gridG.clear();
        } catch (err) {
          if (
            typeof globalThis.process !== "undefined" &&
            (globalThis.process as any).env &&
            (globalThis.process as any).env.NODE_ENV !== "production"
          )
            console.debug("perceptron: grid clear error", err);
        }
        try {
          computeGridSteps();
        } catch (err) {
          if (
            typeof globalThis.process !== "undefined" &&
            (globalThis.process as any).env &&
            (globalThis.process as any).env.NODE_ENV !== "production"
          )
            console.debug("perceptron: computeGridSteps error", err);
        }
        for (let i = 0; i < 50; i++) {
          const vx = p.random(-1, 1),
            vy = p.random(-1, 1);
          const labelSigned = vx * 0.5 + 0.2 > vy ? 1 : -1;
          points.push({ x: vx, y: vy, labelSigned });
          X.push([vx, vy]);
          y.push(labelSigned === 1 ? 1 : 0);
        }
        try {
          startedAtMs =
            performance && performance.now ? performance.now() : Date.now();
        } catch {
          startedAtMs = Date.now();
        }
        try {
          if (finishedRef.current) finishedRef.current.style.display = "none";
        } catch (err) {
          if (
            typeof globalThis.process !== "undefined" &&
            (globalThis.process as any).env &&
            (globalThis.process as any).env.NODE_ENV !== "production"
          )
            console.debug("perceptron: finishedRef hide error", err);
        }
      };

      p.updateDataset = (newDataset: Point[] | undefined) => {
        points = [];
        X = [];
        y = [];
        if (newDataset && Array.isArray(newDataset) && newDataset.length) {
          for (const pt of newDataset) {
            const vx = pt.x,
              vy = pt.y;
            const labelSigned = pt.label === "A" ? 1 : -1;
            points.push({ x: vx, y: vy, labelSigned });
            X.push([vx, vy]);
            y.push(labelSigned === 1 ? 1 : 0);
          }
        }
        model = adaptModel(createRawModel());
        epoch = indexInEpoch = _errorsInEpoch = 0;
        zeroEpochStreak = 0;
        mistakesThisEpoch = 0;
        notifiedDone = false;
        lastMaximizeSig = null;
        lastBestModel = null;
        paused = false;
        trainingAccumulator = 0;
        wasPaused = false;
        try {
          if (gridG) gridG.clear();
        } catch (err) {
          if (
            typeof globalThis.process !== "undefined" &&
            (globalThis.process as any).env &&
            (globalThis.process as any).env.NODE_ENV !== "production"
          )
            console.debug("perceptron: grid clear error", err);
        }
        try {
          computeGridSteps();
        } catch (err) {
          if (
            typeof globalThis.process !== "undefined" &&
            (globalThis.process as any).env &&
            (globalThis.process as any).env.NODE_ENV !== "production"
          )
            console.debug("perceptron: computeGridSteps error", err);
        }
        try {
          startedAtMs =
            performance && performance.now ? performance.now() : Date.now();
        } catch {
          startedAtMs = Date.now();
        }
        try {
          if (finishedRef.current) finishedRef.current.style.display = "none";
        } catch (err) {
          if (
            typeof globalThis.process !== "undefined" &&
            (globalThis.process as any).env &&
            (globalThis.process as any).env.NODE_ENV !== "production"
          )
            console.debug("perceptron: finishedRef hide error", err);
        }
      };

      const safeTrainSample = (sample: any, label: any) => {
        try {
          model.trainSample(sample as number[], label as number);
        } catch {
          // Ignore training errors
          model = adaptModel(createRawModel());
        }
      };
      const safePredict = (sample: any) => {
        try {
          return model.predict(sample as number[]);
        } catch {
          // Ignore prediction errors
          model = adaptModel(createRawModel());
          return 0;
        }
      };

      const safePredictRaw = (sample: any) => {
        try {
          return model.predictRaw(sample as number[]);
        } catch {
          // Ignore prediction errors
          model = adaptModel(createRawModel());
          return 0;
        }
      };
      const computeGridSteps = () => {
        const minDim = Math.max(1, Math.min(p.width || 600, p.height || 600));
        const base = Math.max(4, Math.floor(minDim / 90)); // ~90 cells on short side
        fineGridStep = base; // fine when paused
        coarseGridStep = Math.max(8, base * 2); // coarse while training
        if (
          externalDataset &&
          Array.isArray(externalDataset) &&
          externalDataset.length > 150
        ) {
          fineGridStep = Math.max(fineGridStep, 8);
          coarseGridStep = Math.max(coarseGridStep, 16);
        }
      };

      p.setup = () => {
        if (sketchRef.current) {
          sketchRef.current
            .querySelectorAll("canvas")
            .forEach((c) => c.remove());
        }
        const rect = sketchRef.current?.getBoundingClientRect();
        const w = Math.max(300, rect?.width || 600);
        const h = Math.max(300, rect?.height || 600);
        p.createCanvas(w, h);
        try {
          p.frameRate?.(45);
        } catch (err) {
          if (
            typeof globalThis.process !== "undefined" &&
            (globalThis.process as any).env &&
            (globalThis.process as any).env.NODE_ENV !== "production"
          )
            console.debug("perceptron: frameRate error", err);
        }
        try {
          p.pixelDensity?.(1);
        } catch (err) {
          if (
            typeof globalThis.process !== "undefined" &&
            (globalThis.process as any).env &&
            (globalThis.process as any).env.NODE_ENV !== "production"
          )
            console.debug("perceptron: pixelDensity error", err);
        }
        gridG = p.createGraphics
          ? (p.createGraphics(p.width, p.height) as unknown as P5Graphics)
          : null;
        computeGridSteps();
        if (sketchRef.current) {
          sketchRef.current
            .querySelectorAll(".pd-fallback, .pd-status")
            .forEach((el) => el.remove());
        }
        if (equationRef.current) equationRef.current.textContent = "";
        reset();
        if (
          externalDataset &&
          Array.isArray(externalDataset) &&
          externalDataset.length
        ) {
          p.updateDataset(externalDataset);
        }

        p.maximizeMargin = (options: any = {}) => {
          try {
            const mmStart =
              performance && performance.now ? performance.now() : Date.now();
            if (!Array.isArray(X) || X.length === 0) {
              return;
            }
            // Hyperparameter sweeps + multi-restarts to find a strong separator in one click
            const base = {
              epochs: 120,
              lr: 0.01,
              lambda: 0.001,
              shuffle: true,
              ...options,
            } as {
              epochs: number;
              lr: number;
              lambda: number;
              shuffle: boolean;
            };
            const lrs = options.lr ? [options.lr] : [0.005, 0.01, 0.02, 0.04];
            const lambdas = options.lambda
              ? [options.lambda]
              : [0.0005, 0.001, 0.003];
            const epochs = Math.max(80, base.epochs);
            const restarts = 3; // random init restarts per combo

            type Cand = { err: number; hinge: number; model: any };
            let best: Cand | null = null;

            // Build dataset signature to cache best results for this exact data
            const sig = (() => {
              try {
                const n = X.length;
                const lab = (y || [])
                  .map((v: any) => (v === 1 ? "1" : "0"))
                  .join("");
                const coords = X.slice(0, 120)
                  .map(
                    (row: number[]) =>
                      `${Number(row[0]).toFixed(3)},${Number(row[1]).toFixed(
                        3
                      )}`
                  )
                  .join("|");
                return `${n}|${lab}|${coords}`;
              } catch (err) {
                if (isDev()) console.debug("perceptron: sig build error", err);
                return `n${X && X.length}`;
              }
            })();

            if (lastMaximizeSig && sig === lastMaximizeSig && lastBestModel) {
              // Reuse best model from cache for identical dataset
              model = adaptModel(lastBestModel);
              paused = true;
              notifiedDone = true;
              try {
                if (gridG) gridG.clear();
              } catch (err) {
                if (isDev()) console.debug("perceptron: grid clear error", err);
              }
              // Calculate accuracy
              let accuracy = 1.0;
              try {
                const wrappedLast = adaptModel(lastBestModel);
                if (typeof wrappedLast.misclassificationRate === "function") {
                  const errorRate = wrappedLast.misclassificationRate(X, y);
                  accuracy = 1.0 - errorRate;
                }
              } catch (err) {
                if (
                  typeof globalThis.process !== "undefined" &&
                  (globalThis.process as any).env &&
                  (globalThis.process as any).env.NODE_ENV !== "production"
                )
                  console.debug("perceptron: accuracy calc error", err);
              }
              try {
                const mmElapsed =
                  (performance && performance.now
                    ? performance.now()
                    : Date.now()) - mmStart;
                const elapsedSec = mmElapsed / 1000;
                window.dispatchEvent(
                  new CustomEvent("mlv:demo-finished", {
                    detail: {
                      classifier:
                        classifierType === "poly"
                          ? "Polynomial Perceptron"
                          : "Linear Perceptron",
                      reason: "max-margin",
                      accuracy,
                      elapsedSec,
                    },
                  })
                );
                if (finishedRef.current) {
                  finishedRef.current.innerText = `Max margin finished — Time: ${elapsedSec.toFixed(
                    2
                  )}s`;
                  finishedRef.current.style.display = "block";
                }
              } catch (err) {
                if (isDev()) console.debug("perceptron: event/DOM error", err);
              }
              return;
            }

            const evalCandidate = (m: unknown): Cand => {
              // Both Perceptron and PolynomialPerceptron expose these metrics; wrap via adaptModel
              let err = Number.POSITIVE_INFINITY;
              let hinge = Number.POSITIVE_INFINITY;
              try {
                const wrapped = adaptModel(m);
                err =
                  typeof wrapped.misclassificationRate === "function"
                    ? wrapped.misclassificationRate(X, y)
                    : err;
              } catch (e) {
                if (isDev())
                  console.debug("perceptron: misclassificationRate error", e);
              }
              try {
                const wrapped = adaptModel(m);
                hinge =
                  typeof wrapped.hingeLoss === "function"
                    ? wrapped.hingeLoss(X, y)
                    : hinge;
              } catch (e) {
                if (isDev()) console.debug("perceptron: hingeLoss error", e);
              }
              return {
                err: isFinite(err) ? err : 1e9,
                hinge: isFinite(hinge) ? hinge : 1e9,
                model: m,
              };
            };

            for (const lr of lrs) {
              for (const lambda of lambdas) {
                for (let r = 0; r < restarts; r++) {
                  const m = createRawModel();
                  try {
                    const mWrapped = adaptModel(m);
                    if (typeof mWrapped.fitHingeSGD === "function")
                      mWrapped.fitHingeSGD(X, y, {
                        epochs,
                        lr,
                        lambda,
                        shuffle: true,
                      });
                  } catch (e) {
                    if (isDev())
                      console.debug("perceptron: fitHingeSGD error", e);
                  }
                  const cand = evalCandidate(m);
                  if (
                    !best ||
                    cand.err < best.err - 1e-9 ||
                    (Math.abs(cand.err - best.err) < 1e-9 &&
                      cand.hinge < best.hinge)
                  ) {
                    best = cand;
                  }
                }
              }
            }

            if (best && best.model) {
              // Replace current model with best candidate and pause training so result persists
              model = adaptModel(best.model);
              lastMaximizeSig = sig;
              lastBestModel = best.model;
              paused = true;
              notifiedDone = true; // don't auto-retrain/celebrate
              // Force a background repaint once to reflect the new boundary
              try {
                if (gridG) gridG.clear();
              } catch (err) {
                if (isDev()) console.debug("perceptron: grid clear error", err);
              }
              // Calculate accuracy
              let accuracy = 1.0;
              try {
                const wrappedBest = adaptModel(best.model);
                if (typeof wrappedBest.misclassificationRate === "function") {
                  const errorRate = wrappedBest.misclassificationRate(X, y);
                  accuracy = 1.0 - errorRate;
                }
              } catch (err) {
                if (
                  typeof globalThis.process !== "undefined" &&
                  (globalThis.process as any).env &&
                  (globalThis.process as any).env.NODE_ENV !== "production"
                )
                  console.debug("perceptron: accuracy calc error", err);
              }
              // Notify the app so success animation/toast/equation bar update
              try {
                const mmElapsed =
                  (performance && performance.now
                    ? performance.now()
                    : Date.now()) - mmStart;
                const elapsedSec = mmElapsed / 1000;
                window.dispatchEvent(
                  new CustomEvent("mlv:demo-finished", {
                    detail: {
                      classifier:
                        classifierType === "poly"
                          ? "Polynomial Perceptron"
                          : "Linear Perceptron",
                      reason: "max-margin",
                      accuracy,
                      elapsedSec,
                    },
                  })
                );
                if (finishedRef.current) {
                  finishedRef.current.innerText = `Max margin finished — Time: ${elapsedSec.toFixed(
                    2
                  )}s`;
                  finishedRef.current.style.display = "block";
                }
              } catch (err) {
                if (isDev()) console.debug("perceptron: event/DOM error", err);
              }
            }
          } catch (e) {
            console.error("maximizeMargin error", e);
          }
        };
      };

      p.windowResized = () => {
        const rect = sketchRef.current?.getBoundingClientRect();
        p.resizeCanvas(
          Math.max(300, rect?.width || 600),
          Math.max(300, rect?.height || 600)
        );
        gridG = p.createGraphics
          ? (p.createGraphics(p.width, p.height) as unknown as P5Graphics)
          : null;
        computeGridSteps();
      };

      p.draw = () => {
        // Skip heavy updates if tab is hidden
        try {
          if (typeof document !== "undefined" && (document as any).hidden) {
            return;
          }
        } catch (err) {
          if (isDev()) console.debug("perceptron: visibility probe error", err);
        }
        if (!paused && X.length && epoch < MAX_EPOCHS) {
          const effectiveSpeed = speed * (p.speedScale || 1); // Training speed
          trainingAccumulator += effectiveSpeed;
          let steps = Math.floor(trainingAccumulator);
          if (steps > 0) {
            trainingAccumulator -= steps;
            // Limit training steps per frame to prevent blocking
            steps = Math.min(steps, 5);
            for (let s = 0; s < steps; s++) {
              const xi = X[indexInEpoch],
                yi = y[indexInEpoch];
              // compute raw score once and reuse to avoid double dot products
              const score = safePredictRaw(xi);
              const pred01 = score >= 0 ? 1 : 0;
              if (pred01 !== yi) {
                _errorsInEpoch++;
                mistakesThisEpoch++;
              }
              try {
                const raw: any = (model.raw as any) || (model as any);
                if (raw && typeof raw.trainSampleScore01 === "function") {
                  raw.trainSampleScore01(xi, yi, score);
                } else if (raw && typeof raw.trainSampleRaw01 === "function") {
                  raw.trainSampleRaw01(xi, yi);
                } else {
                  safeTrainSample(xi, yi);
                }
              } catch {
                safeTrainSample(xi, yi);
              }
              indexInEpoch++;
              if (indexInEpoch >= X.length) {
                epoch++;
                indexInEpoch = 0;
                _errorsInEpoch = 0;
                // Check for convergence: zero mistakes this epoch
                if (mistakesThisEpoch === 0) {
                  zeroEpochStreak++;
                  // Converged if we have 2 consecutive zero-mistake epochs, or 1 zero-mistake epoch after epoch 5
                  const hasConverged =
                    (zeroEpochStreak >= 2 && epoch >= 2) ||
                    (zeroEpochStreak >= 1 && epoch >= 5);
                  if (hasConverged && !paused && !notifiedDone) {
                    paused = true;
                    notifiedDone = true;
                    finishedAtMs =
                      performance && performance.now
                        ? performance.now()
                        : Date.now();
                    try {
                      // publish latest equation before notifying
                      try {
                        const eq = getCurrentEquation();
                        window.mlvStatus = {
                          classifier: classifierType,
                          equation: eq,
                          weights: model.weights,
                          bias: model.bias,
                          updatedAt: Date.now(),
                        };
                      } catch (err) {
                        if (isDev())
                          console.debug(
                            "perceptron: candidate eval error",
                            err
                          );
                      }
                      // Calculate accuracy
                      let accuracy = 1.0;
                      try {
                        if (typeof model.misclassificationRate === "function") {
                          const errorRate = model.misclassificationRate!(X, y);
                          accuracy = 1.0 - errorRate;
                        }
                      } catch (err) {
                        if (isDev())
                          console.debug(
                            "perceptron: candidate eval error",
                            err
                          );
                      }
                      const elapsedSec =
                        startedAtMs != null
                          ? (finishedAtMs - startedAtMs) / 1000
                          : undefined;
                      window.dispatchEvent(
                        new CustomEvent("mlv:demo-finished", {
                          detail: {
                            classifier:
                              classifierType === "poly"
                                ? "Polynomial Perceptron"
                                : "Linear Perceptron",
                            reason: "converged",
                            epoch,
                            equation: getCurrentEquation(),
                            accuracy,
                            elapsedSec,
                          },
                        })
                      );
                      try {
                        if (finishedRef.current) {
                          const msg = `Training finished — Time: ${(
                            elapsedSec ?? 0
                          ).toFixed(2)}s`;
                          finishedRef.current.innerText = msg;
                          finishedRef.current.style.display = "block";
                        }
                      } catch (err) {
                        if (isDev())
                          console.debug(
                            "perceptron: candidate eval error",
                            err
                          );
                      }
                    } catch {
                      /* ignore potential event/DOM errors */
                    }
                    // banner removed; App now shows equation under the canvas
                  }
                } else {
                  zeroEpochStreak = 0;
                }
                // reset counter for next epoch
                mistakesThisEpoch = 0;
                if (epoch >= MAX_EPOCHS && !notifiedDone) {
                  paused = true;
                  notifiedDone = true;
                  finishedAtMs =
                    performance && performance.now
                      ? performance.now()
                      : Date.now();
                  try {
                    try {
                      const eq = getCurrentEquation();
                      (window as any).mlvStatus = {
                        classifier: classifierType,
                        equation: eq,
                        weights: model.weights,
                        bias: model.bias,
                        updatedAt: Date.now(),
                      };
                    } catch (err) {
                      if (isDev())
                        console.debug("perceptron: event/DOM error", err);
                    }
                    // Calculate accuracy
                    let accuracy = 1.0;
                    try {
                      if (typeof model.misclassificationRate === "function") {
                        const errorRate = model.misclassificationRate!(X, y);
                        accuracy = 1.0 - errorRate;
                      }
                    } catch (err) {
                      if (isDev())
                        console.debug("perceptron: event/DOM error", err);
                    }
                    const elapsedSec =
                      startedAtMs != null
                        ? (finishedAtMs - startedAtMs) / 1000
                        : undefined;
                    window.dispatchEvent(
                      new CustomEvent("mlv:demo-finished", {
                        detail: {
                          classifier:
                            classifierType === "poly"
                              ? "Polynomial Perceptron"
                              : "Linear Perceptron",
                          reason: "max-epochs",
                          epoch,
                          equation: getCurrentEquation(),
                          accuracy,
                          elapsedSec,
                        },
                      })
                    );
                    try {
                      if (finishedRef.current) {
                        const msg = `Training finished — Time: ${(
                          elapsedSec ?? 0
                        ).toFixed(2)}s`;
                        finishedRef.current.innerText = msg;
                        finishedRef.current.style.display = "block";
                      }
                    } catch (err) {
                      if (isDev())
                        console.debug("perceptron: DOM update error", err);
                    }
                  } catch {
                    /* ignore paint errors */
                  }
                  // banner removed; App now shows equation under the canvas
                }
              }
            }
          }
        }

        // cache color scheme once per frame
        const isDark =
          window.matchMedia &&
          window.matchMedia("(prefers-color-scheme: dark)").matches;
        // Celebration grace window: skip heavy redraws for a short time after finish so confetti shows instantly
        const celebrateGrace = () => {
          const now =
            performance && performance.now ? performance.now() : Date.now();
          return notifiedDone && finishedAtMs && now - finishedAtMs < 500; // 0.5s grace
        };
        p.background(isDark ? 30 : 255);

        if (classifierType === "poly") {
          // Grid-based fill for non-linear boundaries
          // During celebration grace, keep coarse grid to avoid heavy recompute immediately
          const inGrace = celebrateGrace();
          gridStep = paused && !inGrace ? fineGridStep : coarseGridStep;
          gridFreq = paused && !inGrace ? 60 : 18;
          if (!gridG)
            gridG = p.createGraphics
              ? (p.createGraphics(p.width, p.height) as unknown as P5Graphics)
              : null;
          if (wasPaused !== paused) {
            wasPaused = paused;
            // Avoid clearing immediately on finish; let existing grid persist under confetti
            if (!inGrace)
              try {
                gridG.clear();
              } catch (err) {
                if (
                  typeof globalThis.process !== "undefined" &&
                  (globalThis.process as any).env &&
                  (globalThis.process as any).env.NODE_ENV !== "production"
                )
                  console.debug("perceptron: grid clear error", err);
              }
          }
          if (
            !inGrace &&
            (p.frameCount % gridFreq === 0 || wasPaused !== paused)
          ) {
            gridG.clear();
            const step = Math.max(2, gridStep);
            gridG.noStroke();
            for (let px = 0; px < p.width; px += step) {
              for (let py = 0; py < p.height; py += step) {
                const vx = p.map(px + step / 2, 0, p.width, -1, 1);
                const vy = p.map(py + step / 2, p.height, 0, -1, 1);
                const pred = safePredict([vx, vy]);
                gridG.fill(
                  pred === 1
                    ? isDark
                      ? "rgba(80,150,255,0.3)"
                      : "rgba(200,230,255,0.9)"
                    : isDark
                    ? "rgba(255,100,100,0.3)"
                    : "rgba(255,220,220,0.9)"
                );
                const rectWidth = Math.min(step, p.width - px);
                const rectHeight = Math.min(step, p.height - py);
                gridG.rect(px, py, rectWidth, rectHeight);
              }
            }
          }
          p.image(gridG, 0, 0);
        } else if (classifierType === "linear") {
          // Smooth, straight separation without grid: repaint background only when needed
          if (!gridG)
            gridG = p.createGraphics
              ? (p.createGraphics(p.width, p.height) as unknown as P5Graphics)
              : null;
          const classA = isDark
            ? "rgba(80,150,255,0.3)"
            : "rgba(200,230,255,0.9)";
          const classB = isDark
            ? "rgba(255,100,100,0.3)"
            : "rgba(255,220,220,0.9)";
          const inGrace = celebrateGrace();
          // Keep coarse fill during grace to avoid a full-width repaint in the finish frame
          const colStep = paused && !inGrace ? 1 : 2; // finer when fully paused, coarse during grace
          const wts = model.weights || [];
          const b = model.bias || 0;

          // Track last rendered state to avoid per-frame recomputation
          pEx._linLast = pEx._linLast || {
            w0: NaN,
            w1: NaN,
            b: NaN,
            colStep: NaN,
            w: 0,
            h: 0,
          };
          const last = pEx._linLast;
          const w0 = wts[0] ?? NaN;
          const w1 = wts[1] ?? NaN;
          const sizeChanged = last.w !== p.width || last.h !== p.height;
          const colChanged = last.colStep !== colStep;
          const weightsChanged = !(
            Math.abs(last.w0 - w0) < 1e-6 &&
            Math.abs(last.w1 - w1) < 1e-6 &&
            Math.abs(last.b - b) < 1e-6
          );

          const shouldRepaint =
            (sizeChanged || colChanged || weightsChanged) && !inGrace;
          const throttle = paused ? 1 : 0; // repaint immediately when paused, otherwise throttle via frame cadence below
          if (shouldRepaint && (throttle || p.frameCount % 6 === 0)) {
            gridG.clear();
            gridG.noStroke();
            if (wts.length >= 2) {
              if (Math.abs(w1) > 1e-8) {
                for (let x = 0; x < p.width; x += colStep) {
                  const nx = p.map(x + colStep / 2, 0, p.width, -1, 1);
                  const yNorm = -(w0 * nx + b) / w1; // boundary y in normalized coords
                  const yPix = p.map(yNorm, -1, 1, p.height, 0);
                  // sample a point just above and just below the boundary to decide colors
                  const yTopNorm = p.map(
                    Math.max(0, yPix - 1),
                    p.height,
                    0,
                    -1,
                    1
                  );
                  const yBotNorm = p.map(
                    Math.min(p.height, yPix + 1),
                    p.height,
                    0,
                    -1,
                    1
                  );
                  const predTop = safePredict([nx, yTopNorm]) === 1;
                  const predBot = safePredict([nx, yBotNorm]) === 1;
                  const topH = Math.max(0, Math.min(p.height, yPix));
                  const botY = Math.max(0, Math.min(p.height, yPix));
                  if (topH > 0) {
                    gridG.fill(predTop ? classA : classB);
                    gridG.rect(x, 0, colStep, topH);
                  }
                  if (botY < p.height) {
                    gridG.fill(predBot ? classA : classB);
                    gridG.rect(x, botY, colStep, p.height - botY);
                  }
                }
              } else if (Math.abs(w0) > 1e-8) {
                // Vertical boundary: x = -b / w0
                const xNorm = -b / w0;
                const xPix = p.map(xNorm, -1, 1, 0, p.width);
                // sample left and right
                const leftPred =
                  safePredict([
                    p.map(Math.max(0, xPix - 1), 0, p.width, -1, 1),
                    0,
                  ]) === 1;
                const rightPred =
                  safePredict([
                    p.map(Math.min(p.width, xPix + 1), 0, p.width, -1, 1),
                    0,
                  ]) === 1;
                const leftW = Math.max(0, Math.min(p.width, xPix));
                const rightX = Math.max(0, Math.min(p.width, xPix));
                if (leftW > 0) {
                  gridG.fill(leftPred ? classA : classB);
                  gridG.rect(0, 0, leftW, p.height);
                }
                if (rightX < p.width) {
                  gridG.fill(rightPred ? classA : classB);
                  gridG.rect(rightX, 0, p.width - rightX, p.height);
                }
              } else {
                // Degenerate: fill with majority prediction at center
                const centerPred = safePredict([0, 0]) === 1;
                gridG.fill(centerPred ? classA : classB);
                gridG.rect(0, 0, p.width, p.height);
              }
            }
            // record last rendered state
            last.w0 = w0;
            last.w1 = w1;
            last.b = b;
            last.colStep = colStep;
            last.w = p.width;
            last.h = p.height;
          }
          p.image(gridG, 0, 0);
        }

        const weights = model.weights || [];
        const bias = model.bias || 0;

        // Publish equation to global for AgentPanel
        try {
          if (classifierType === "linear") {
            const eq =
              weights.length >= 2
                ? buildLinearEquation(weights, bias)
                : "linear: insufficient weights";
            (window as any).mlvStatus = {
              classifier: "linear",
              equation: eq,
              weights,
              bias,
              updatedAt: Date.now(),
            };
          } else if (classifierType === "poly") {
            const eq = buildPolyEquation(weights, bias);
            (window as any).mlvStatus = {
              classifier: "poly",
              equation: eq,
              weights,
              bias,
              updatedAt: Date.now(),
            };
          }
        } catch {
          /* ignore top-level draw errors */
        }

        if (
          classifierType === "linear" &&
          notifiedDone &&
          weights.length >= 2
        ) {
          p.stroke(isDark ? 255 : 0, 150, 255);
          const w0 = weights[0];
          const w1 = weights[1];
          if (Math.abs(w1) > 1e-6) {
            const x1 = -1,
              x2 = 1;
            const y1 = -(w0 * x1 + bias) / w1;
            const y2 = -(w0 * x2 + bias) / w1;
            p.line(
              p.map(x1, -1, 1, 0, p.width),
              p.map(y1, -1, 1, p.height, 0),
              p.map(x2, -1, 1, 0, p.width),
              p.map(y2, -1, 1, p.height, 0)
            );
          } else if (Math.abs(w0) > 1e-6) {
            // vertical boundary: x = -b / w0
            const xNorm = -bias / w0;
            const xPix = p.map(xNorm, -1, 1, 0, p.width);
            p.line(xPix, 0, xPix, p.height);
          }
        }

        for (let pt of points) {
          const px = p.map(pt.x, -1, 1, 0, p.width);
          const py = p.map(pt.y, -1, 1, p.height, 0);
          const predSigned = safePredict([pt.x, pt.y]) === 1 ? 1 : -1;
          p.fill(
            pt.labelSigned === 1
              ? isDark
                ? "cyan"
                : "blue"
              : isDark
              ? "orange"
              : "red"
          );
          p.stroke(0);
          p.circle(px, py, 10);
          if (predSigned !== pt.labelSigned) {
            p.noFill();
            p.stroke(0);
            p.circle(px, py, 14);
          }
        }

        let showEquation = false;
        const mouseXNorm = p.map(p.mouseX, 0, p.width, -1, 1);
        const mouseYNorm = p.map(p.mouseY, p.height, 0, -1, 1);
        if (
          classifierType === "linear" &&
          notifiedDone &&
          weights.length >= 2 &&
          Math.abs(weights[1]) > 1e-6
        ) {
          const yOnLine = -(weights[0] * mouseXNorm + bias) / weights[1];
          if (Math.abs(yOnLine - mouseYNorm) < 0.05) showEquation = true;
        } else if (classifierType === "poly" && notifiedDone) {
          if (Math.abs(safePredict([mouseXNorm, mouseYNorm]) - 0.5) < 0.1)
            showEquation = true;
        }

        if (equationRef.current) {
          equationRef.current.style.opacity = showEquation ? "1" : "0";
          if (showEquation) {
            equationRef.current.textContent =
              classifierType === "linear"
                ? buildLinearEquation(weights, bias)
                : buildPolyEquation(weights, bias);
          }
        }
      };

      p.keyPressed = () => {
        const mouseXNorm = p.map(p.mouseX, 0, p.width, -1, 1);
        const mouseYNorm = p.map(p.mouseY, p.height, 0, -1, 1);
        if (p.key === " ") {
          // Only toggle pause/resume; do not mark as finished or show overlays on manual pause
          paused = !paused;
          return;
        }
        if (p.key === "b") {
          points.push({ x: mouseXNorm, y: mouseYNorm, labelSigned: 1 });
          X.push([mouseXNorm, mouseYNorm]);
          y.push(1);
          safeTrainSample([mouseXNorm, mouseYNorm], 1);
          if (onDatasetChange)
            onDatasetChange((d: any) => [
              ...d,
              { x: mouseXNorm, y: mouseYNorm, label: "A" },
            ]);
        }
        if (p.key === "r") {
          points.push({ x: mouseXNorm, y: mouseYNorm, labelSigned: -1 });
          X.push([mouseXNorm, mouseYNorm]);
          y.push(0);
          safeTrainSample([mouseXNorm, mouseYNorm], 0);
          if (onDatasetChange)
            onDatasetChange((d: any) => [
              ...d,
              { x: mouseXNorm, y: mouseYNorm, label: "B" },
            ]);
        }
        if (p.key === "+" || p.key === "=") speed = Math.min(20, speed + 1);
        if (p.key === "-" || p.key === "_") speed = Math.max(0, speed - 1);
      };

      p.resetDemo = () => {
        try {
          if (onDatasetChange) onDatasetChange([]);
        } catch {
          // Ignore callback errors
        }
        // Always fully reset the local demo state to ensure training restarts
        reset();
        try {
          window.dispatchEvent(
            new CustomEvent("mlv:demo-reset", {
              detail: { classifier: classifierType },
            })
          );
        } catch (err) {
          if (
            typeof globalThis.process !== "undefined" &&
            (globalThis.process as any).env &&
            (globalThis.process as any).env.NODE_ENV !== "production"
          )
            console.debug("perceptron: event dispatch error", err);
        }
        try {
          if (finishedRef.current) finishedRef.current.style.display = "none";
        } catch (err) {
          if (
            typeof globalThis.process !== "undefined" &&
            (globalThis.process as any).env &&
            (globalThis.process as any).env.NODE_ENV !== "production"
          )
            console.debug("perceptron: finishedRef hide error", err);
        }
      };
    };

    let mounted = true;
    setLoading(true);
    (async () => {
      try {
        const mod = await import("p5");
        if (!mounted) return;
        const P5 = (mod && (mod as any).default) || mod;
        if (p5InstanceRef.current) {
          p5InstanceRef.current.remove();
          p5InstanceRef.current = null;
        }
        if (sketchRef.current) {
          p5InstanceRef.current = new P5(
            sketch,
            sketchRef.current
          ) as unknown as P5Instance;
        }
        setLoading(false);
      } catch (err) {
        if (isDev()) console.debug("perceptron: dynamic import p5 failed", err);
        setLoading(false);
      }
    })();
    return () => {
      mounted = false;
      if (p5InstanceRef.current) {
        p5InstanceRef.current.remove();
        p5InstanceRef.current = null;
      }
    };
  }, [classifierType, externalDataset, onDatasetChange]);

  // NOTE: dataset updates are handled by the reset-token-aware effect below which
  // listens for both `resetToken` and `externalDataset`. Keeping a single effect
  // reduces duplicate calls into the drawn p5 instance and avoids unnecessary
  // re-render/update churn.

  useEffect(() => {
    if (
      p5InstanceRef.current &&
      typeof p5InstanceRef.current.updateDataset === "function"
    ) {
      p5InstanceRef.current.updateDataset(externalDataset || []);
    }
  }, [resetToken, externalDataset]);

  useEffect(() => {
    if (p5InstanceRef.current) {
      p5InstanceRef.current.speedScale = speedScale || 1;
    }
  }, [speedScale]);

  const handleReset = React.useCallback(() => {
    try {
      if (onDatasetChange) onDatasetChange([]);
    } catch {
      // Ignore callback errors
    }
    if (p5InstanceRef.current) {
      if (typeof p5InstanceRef.current.resetDemo === "function") {
        p5InstanceRef.current.resetDemo();
      } else if (typeof p5InstanceRef.current.updateDataset === "function") {
        p5InstanceRef.current.updateDataset([]);
      }
    }
  }, [onDatasetChange]);

  const handleMaximizeMargin = React.useCallback(() => {
    if (
      p5InstanceRef.current &&
      typeof p5InstanceRef.current.maximizeMargin === "function"
    ) {
      p5InstanceRef.current.maximizeMargin();
    }
  }, []);

  return (
    <div ref={sketchRef} className={styles.container}>
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
      <button
        type="button"
        id="perceptron-reset"
        onClick={handleReset}
        className={`${styles.btn} ${styles.btnLeft}`}
      >
        Reset
      </button>
      <button
        type="button"
        id="perceptron-maximize"
        onClick={handleMaximizeMargin}
        className={`${styles.btn} ${styles.btnRight}`}
      >
        Maximize margin
      </button>
      <div ref={equationRef} className={styles.equation} />
      {/* finished time overlay */}
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
          zIndex: 1500,
          boxShadow: "0 4px 12px rgba(0,0,0,0.25)",
        }}
      >
        Training finished
      </div>
      {/* Removed the on-canvas finished banner; App renders equation beneath the canvas */}
    </div>
  );
};

export default PerceptronDemo;
