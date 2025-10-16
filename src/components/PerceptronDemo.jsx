import React, { useEffect, useRef } from "react";
import p5 from "p5";
import Perceptron from "../utils/perceptron";
import PolynomialPerceptron from "../utils/polynomialClassifier";

const PerceptronDemo = ({
  classifierType = "linear",
  dataset: externalDataset,
  onDatasetChange,
  resetToken,
}) => {
  const sketchRef = useRef();
  const p5InstanceRef = useRef();
  const equationRef = useRef();

  useEffect(() => {
    const setStatus = (msg) => {
      if (sketchRef.current) {
        const s = sketchRef.current.querySelector(".pd-status");
        if (s) s.textContent = msg;
      }
    };

    const fmt = (n) =>
      !isFinite(n) ? "0" : Math.abs(n) < 1e-6 ? 0 : Number(n.toFixed(3));

    const buildLinearEquation = (weights = [], bias = 0) => {
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

    const buildPolyEquation = (weights = [], bias = 0) => {
      const terms = [];
      const names = ["x", "y", "x^2", "y^2", "xy"];
      for (let i = 0; i < Math.min(weights.length, 5); i++) {
        if (Math.abs(weights[i]) < 1e-6) continue;
        terms.push(`${fmt(weights[i])}·${names[i]}`);
      }
      return `${terms.join(" + ") || "0"} ${bias >= 0 ? "+" : "-"} ${fmt(
        Math.abs(bias)
      )} = 0`;
    };

    const sketch = (p) => {
      let points = [],
        X = [],
        y = [];
      let gridG = null; // offscreen graphics for poly decision background
      const defaultGridStep = 8;
      let gridStep = defaultGridStep;
      let gridFreq = 6; // frames between recomputing grid
      let model,
        // when in compare mode (shared dataset) default to not paused
        paused = false,
        // use a responsive default speed in compare mode so training and visuals update
        speed =
          externalDataset &&
          Array.isArray(externalDataset) &&
          externalDataset.length
            ? 1
            : 0.3,
        // accumulator to allow fractional speeds (<1) to produce occasional steps
        trainingAccumulator = 0,
        epoch = 0,
        indexInEpoch = 0,
        errorsInEpoch = 0;
      const MAX_EPOCHS = 1000;

      const createRawModel = () =>
        classifierType === "poly"
          ? new PolynomialPerceptron(2, 0.05, 1)
          : new Perceptron(2, 0.05, 1);

      const adaptModel = (m) => ({
        raw: m,
        predict: (sample) => {
          if (!m) return 0;
          if (typeof m.predict === "function") return m.predict(sample);
          if (m.raw && typeof m.raw.predict === "function")
            return m.raw.predict(sample);
          return 0;
        },
        trainSample: (sample, label) => {
          if (!m) return;
          if (typeof m.trainSample === "function")
            return m.trainSample(sample, label);
          if (typeof m.fit === "function") return m.fit([sample], [label]);
        },
        get weights() {
          return m.weights || (m.raw && m.raw.weights) || [];
        },
        get bias() {
          return m.bias || (m.raw && m.raw.bias) || 0;
        },
      });

      const reset = () => {
        model = adaptModel(createRawModel());
        points = [];
        X = [];
        y = [];
        epoch = indexInEpoch = errorsInEpoch = 0;
        // by default start with random points; external dataset can be applied later via p.updateDataset
        for (let i = 0; i < 50; i++) {
          const vx = p.random(-1, 1),
            vy = p.random(-1, 1);
          const labelSigned = vx * 0.5 + 0.2 > vy ? 1 : -1;
          points.push({ x: vx, y: vy, labelSigned });
          X.push([vx, vy]);
          y.push(labelSigned === 1 ? 1 : 0);
        }
      };

      // allow external updates to dataset without recreating the sketch
      p.updateDataset = (newDataset) => {
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
        // reset model and training state to learn from new dataset
        model = adaptModel(createRawModel());
        epoch = indexInEpoch = errorsInEpoch = 0;
      };

      const safeTrainSample = (sample, label) => {
        try {
          model.trainSample(sample, label);
        } catch (e) {
          model = adaptModel(createRawModel());
        }
      };
      const safePredict = (sample) => {
        try {
          return model.predict(sample);
        } catch (e) {
          model = adaptModel(createRawModel());
          return 0;
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
        gridG = p.createGraphics(p.width, p.height);
        if (sketchRef.current) {
          sketchRef.current
            .querySelectorAll(".pd-fallback, .pd-status")
            .forEach((el) => el.remove());
        }
        if (equationRef.current) equationRef.current.textContent = "";
        reset();
        // if a shared dataset was provided, initialize from it immediately
        if (
          externalDataset &&
          Array.isArray(externalDataset) &&
          externalDataset.length
        ) {
          p.updateDataset(externalDataset);
        }

        // expose a helper to run hinge-SGD training to maximize margin
        p.maximizeMargin = (options = {}) => {
          try {
            const opts = Object.assign(
              { epochs: 150, lr: 0.01, lambda: 0.001, shuffle: true },
              options
            );
            if (!X || !X.length) return;
            // model.raw is the underlying Perceptron/PolynomialPerceptron's raw Perceptron
            if (
              model &&
              model.raw &&
              typeof model.raw.fitHingeSGD === "function"
            ) {
              model.raw.fitHingeSGD(X, y, opts);
            } else if (model && typeof model.fitHingeSGD === "function") {
              // fallback if adaptModel wrapped differently
              model.fitHingeSGD(X, y, opts);
            }

            // recompute poly grid immediately so the background updates right away
            if (classifierType === "poly" && gridG) {
              gridG.clear();
              const darkMode =
                window.matchMedia &&
                window.matchMedia("(prefers-color-scheme: dark)").matches;
              const step = gridStep || defaultGridStep;
              for (let px = 0; px < p.width; px += step) {
                for (let py = 0; py < p.height; py += step) {
                  const vx = p.map(px + step / 2, 0, p.width, -1, 1);
                  const vy = p.map(py + step / 2, p.height, 0, -1, 1);
                  const pred = safePredict([vx, vy]);
                  gridG.noStroke();
                  gridG.fill(
                    pred === 1
                      ? darkMode
                        ? "rgba(80,150,255,0.3)"
                        : "rgba(200,230,255,0.9)"
                      : darkMode
                      ? "rgba(255,100,100,0.3)"
                      : "rgba(255,220,220,0.9)"
                  );
                  gridG.rect(px, py, step, step);
                }
              }
            }
          } catch (e) {
            // ignore errors during optional optimization
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
        gridG = p.createGraphics(p.width, p.height);
      };

      p.draw = () => {
        // train
        if (!paused && X.length && epoch < MAX_EPOCHS) {
          // allow fractional speed: accumulate and run floor(accumulator) steps
          trainingAccumulator += speed;
          let steps = Math.floor(trainingAccumulator);
          if (steps > 0) {
            trainingAccumulator -= steps;
            for (let s = 0; s < steps; s++) {
              const xi = X[indexInEpoch],
                yi = y[indexInEpoch];
              const predBefore = safePredict(xi);
              if (predBefore !== yi) errorsInEpoch++;
              safeTrainSample(xi, yi);
              indexInEpoch++;
              if (indexInEpoch >= X.length) {
                // advance epoch and reset counters; do not auto-pause on zero errors
                epoch++;
                indexInEpoch = 0;
                errorsInEpoch = 0;
                if (epoch >= MAX_EPOCHS) paused = true;
              }
            }
          }
        }

        // dark/light background
        const darkMode =
          window.matchMedia &&
          window.matchMedia("(prefers-color-scheme: dark)").matches;
        p.background(darkMode ? 30 : 255);

        // draw background grid for poly (use cached offscreen graphics)
        if (classifierType === "poly") {
          // adjust frequency/step when external dataset (likely compare mode)
          if (
            externalDataset &&
            Array.isArray(externalDataset) &&
            externalDataset.length
          ) {
            // use larger tiles and recompute less frequently for speed
            gridStep = Math.max(20, defaultGridStep * 3);
            gridFreq = 60; // recompute infrequently in compare mode
          } else {
            gridStep = defaultGridStep;
            gridFreq = 6;
          }

          if (!gridG) gridG = p.createGraphics(p.width, p.height);

          // recompute grid periodically to reduce CPU
          if (p.frameCount % gridFreq === 0) {
            gridG.clear();
            for (let px = 0; px < p.width; px += gridStep) {
              for (let py = 0; py < p.height; py += gridStep) {
                const vx = p.map(px + gridStep / 2, 0, p.width, -1, 1);
                const vy = p.map(py + gridStep / 2, p.height, 0, -1, 1);
                const pred = safePredict([vx, vy]);
                gridG.noStroke();
                gridG.fill(
                  pred === 1
                    ? darkMode
                      ? "rgba(80,150,255,0.3)"
                      : "rgba(200,230,255,0.9)"
                    : darkMode
                    ? "rgba(255,100,100,0.3)"
                    : "rgba(255,220,220,0.9)"
                );
                gridG.rect(px, py, gridStep, gridStep);
              }
            }
          }

          // draw cached grid
          p.image(gridG, 0, 0);
        }

        const weights = model.weights || [];
        const bias = model.bias || 0;

        // draw line for linear
        if (
          classifierType === "linear" &&
          weights.length >= 2 &&
          Math.abs(weights[1]) > 1e-6
        ) {
          p.stroke(darkMode ? 255 : 0, 150, 255);
          const x1 = -1,
            x2 = 1;
          const y1 = -(weights[0] * x1 + bias) / weights[1];
          const y2 = -(weights[0] * x2 + bias) / weights[1];
          p.line(
            p.map(x1, -1, 1, 0, p.width),
            p.map(y1, -1, 1, p.height, 0),
            p.map(x2, -1, 1, 0, p.width),
            p.map(y2, -1, 1, p.height, 0)
          );
        }

        // draw points
        for (let pt of points) {
          const px = p.map(pt.x, -1, 1, 0, p.width);
          const py = p.map(pt.y, -1, 1, p.height, 0);
          const predSigned = safePredict([pt.x, pt.y]) === 1 ? 1 : -1;
          p.fill(
            pt.labelSigned === 1
              ? darkMode
                ? "cyan"
                : "blue"
              : darkMode
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

        // hover-based equation display
        let showEquation = false;
        const mouseXNorm = p.map(p.mouseX, 0, p.width, -1, 1);
        const mouseYNorm = p.map(p.mouseY, p.height, 0, -1, 1);
        if (
          classifierType === "linear" &&
          weights.length >= 2 &&
          Math.abs(weights[1]) > 1e-6
        ) {
          const yOnLine = -(weights[0] * mouseXNorm + bias) / weights[1];
          if (Math.abs(yOnLine - mouseYNorm) < 0.05) showEquation = true;
        } else if (classifierType === "poly") {
          if (Math.abs(safePredict([mouseXNorm, mouseYNorm]) - 0.5) < 0.1)
            showEquation = true;
        }

        if (equationRef.current) {
          equationRef.current.style.opacity = showEquation ? 1 : 0;
          if (showEquation) {
            equationRef.current.textContent =
              classifierType === "linear"
                ? buildLinearEquation(weights, bias)
                : buildPolyEquation(weights, bias);
          }
        }
      };

      // keyboard controls
      p.keyPressed = () => {
        const mouseXNorm = p.map(p.mouseX, 0, p.width, -1, 1);
        const mouseYNorm = p.map(p.mouseY, p.height, 0, -1, 1);
        if (p.key === " ") {
          paused = !paused;
          return;
        }
        if (p.key === "b") {
          points.push({ x: mouseXNorm, y: mouseYNorm, labelSigned: 1 });
          X.push([mouseXNorm, mouseYNorm]);
          y.push(1);
          safeTrainSample([mouseXNorm, mouseYNorm], 1);
          if (onDatasetChange)
            onDatasetChange((d) => [
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
            onDatasetChange((d) => [
              ...d,
              { x: mouseXNorm, y: mouseYNorm, label: "B" },
            ]);
        }
        if (p.key === "R") {
          if (onDatasetChange) onDatasetChange([]);
          reset();
        }
        if (p.key === "+" || p.key === "=") speed = Math.min(20, speed + 1);
        if (p.key === "-" || p.key === "_") speed = Math.max(0, speed - 1);
      };
    };

    if (p5InstanceRef.current) {
      p5InstanceRef.current.remove();
      p5InstanceRef.current = null;
    }
    if (sketchRef.current) {
      p5InstanceRef.current = new p5(sketch, sketchRef.current);
    }

    return () => {
      if (p5InstanceRef.current) {
        p5InstanceRef.current.remove();
        p5InstanceRef.current = null;
      }
    };
  }, [classifierType, externalDataset, onDatasetChange]);

  // update dataset in-place when externalDataset changes
  useEffect(() => {
    if (
      p5InstanceRef.current &&
      typeof p5InstanceRef.current.updateDataset === "function"
    ) {
      p5InstanceRef.current.updateDataset(externalDataset || []);
    }
  }, [externalDataset]);

  // when resetToken changes, force the sketch to re-apply dataset and restart training
  useEffect(() => {
    if (
      p5InstanceRef.current &&
      typeof p5InstanceRef.current.updateDataset === "function"
    ) {
      p5InstanceRef.current.updateDataset(externalDataset || []);
    }
  }, [resetToken]);

  return (
    <div
      ref={sketchRef}
      style={{
        position: "relative",
        width: "100%",
        height: "100%",
        overflow: "hidden",
      }}
    >
      <button
        onClick={() => {
          if (
            p5InstanceRef.current &&
            typeof p5InstanceRef.current.maximizeMargin === "function"
          ) {
            p5InstanceRef.current.maximizeMargin();
          }
        }}
        style={{
          position: "absolute",
          right: 12,
          top: 12,
          zIndex: 200,
          padding: "6px 10px",
          borderRadius: 8,
        }}
      >
        Maximize margin
      </button>
      <div
        ref={equationRef}
        style={{
          position: "absolute",
          top: "50%",
          left: "50%",
          transform: "translate(-50%, -50%)",
          background: "rgba(255,255,255,0.85)",
          padding: "6px 12px",
          borderRadius: 10,
          fontFamily: "monospace",
          fontSize: 16,
          zIndex: 100,
          boxShadow: "0 4px 14px rgba(0,0,0,0.15)",
          opacity: 0,
          transition: "opacity 0.3s ease",
          pointerEvents: "none",
        }}
      />
    </div>
  );
};

export default PerceptronDemo;
