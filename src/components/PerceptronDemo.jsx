import React, { useEffect, useRef } from "react";
import p5 from "p5";
import Perceptron from "../utils/perceptron";
import PolynomialPerceptron from "../utils/polynomialClassifier";

const PerceptronDemo = ({ classifierType = "linear" }) => {
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

    const fmt = (n) => (!isFinite(n) ? "0" : Math.abs(n) < 1e-6 ? 0 : Number(n.toFixed(3)));

    const buildLinearEquation = (weights = [], bias = 0) => {
      if (weights.length < 2) return "linear: no weights";
      const w0 = weights[0], w1 = weights[1];
      if (Math.abs(w1) > 1e-8) {
        const slope = -w0 / w1;
        const intercept = -bias / w1;
        return `y = ${fmt(slope)} x ${intercept >= 0 ? "+" : "-"} ${fmt(Math.abs(intercept))}`;
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
      return `${terms.join(" + ") || "0"} ${bias >= 0 ? "+" : "-"} ${fmt(Math.abs(bias))} = 0`;
    };

    const sketch = (p) => {
      let points = [], X = [], y = [];
      let model, paused = false, speed = 1, epoch = 0, indexInEpoch = 0, errorsInEpoch = 0;
      const MAX_EPOCHS = 1000;

      const createRawModel = () => classifierType === "poly"
        ? new PolynomialPerceptron(2, 0.05, 1)
        : new Perceptron(2, 0.05, 1);

      const adaptModel = (m) => ({
        raw: m,
        predict: (sample) => {
          if (!m) return 0;
          if (typeof m.predict === "function") return m.predict(sample);
          if (m.raw && typeof m.raw.predict === "function") return m.raw.predict(sample);
          return 0;
        },
        trainSample: (sample, label) => {
          if (!m) return;
          if (typeof m.trainSample === "function") return m.trainSample(sample, label);
          if (typeof m.fit === "function") return m.fit([sample], [label]);
        },
        get weights() { return m.weights || (m.raw && m.raw.weights) || []; },
        get bias() { return m.bias || (m.raw && m.raw.bias) || 0; },
      });

      const reset = () => {
        model = adaptModel(createRawModel());
        points = []; X = []; y = [];
        epoch = indexInEpoch = errorsInEpoch = 0;
        for (let i = 0; i < 50; i++) {
          const vx = p.random(-1, 1), vy = p.random(-1, 1);
          const labelSigned = vx * 0.5 + 0.2 > vy ? 1 : -1;
          points.push({ x: vx, y: vy, labelSigned });
          X.push([vx, vy]);
          y.push(labelSigned === 1 ? 1 : 0);
        }
      };

      const safeTrainSample = (sample, label) => {
        try { model.trainSample(sample, label); } catch (e) { model = adaptModel(createRawModel()); }
      };
      const safePredict = (sample) => {
        try { return model.predict(sample); } catch (e) { model = adaptModel(createRawModel()); return 0; }
      };

      p.setup = () => {
        if (sketchRef.current) {
          sketchRef.current.querySelectorAll("canvas").forEach(c => c.remove());
        }
        const rect = sketchRef.current?.getBoundingClientRect();
        const w = Math.max(300, rect?.width || 600);
        const h = Math.max(300, rect?.height || 600);
        p.createCanvas(w, h);
        if (sketchRef.current) {
          sketchRef.current.querySelectorAll(".pd-fallback, .pd-status").forEach(el => el.remove());
        }
        if (equationRef.current) equationRef.current.textContent = "";
        reset();
      };

      p.windowResized = () => {
        const rect = sketchRef.current?.getBoundingClientRect();
        p.resizeCanvas(Math.max(300, rect?.width || 600), Math.max(300, rect?.height || 600));
      };

      p.draw = () => {
        // train
        if (!paused && X.length && epoch < MAX_EPOCHS) {
          for (let s = 0; s < speed; s++) {
            const xi = X[indexInEpoch], yi = y[indexInEpoch];
            const predBefore = safePredict(xi);
            if (predBefore !== yi) errorsInEpoch++;
            safeTrainSample(xi, yi);
            indexInEpoch++;
            if (indexInEpoch >= X.length) {
              if (errorsInEpoch === 0) { paused = true; break; }
              epoch++; indexInEpoch = 0; errorsInEpoch = 0;
              if (epoch >= MAX_EPOCHS) paused = true;
            }
          }
        }

        // dark/light background
        const darkMode = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
        p.background(darkMode ? 30 : 255);

        // draw background grid for poly
        if (classifierType === "poly") {
          const step = 8;
          for (let px = 0; px < p.width; px += step) {
            for (let py = 0; py < p.height; py += step) {
              const vx = p.map(px + step / 2, 0, p.width, -1, 1);
              const vy = p.map(py + step / 2, p.height, 0, -1, 1);
              const pred = safePredict([vx, vy]);
              p.noStroke();
              p.fill(pred === 1 ? (darkMode ? "rgba(80,150,255,0.3)" : "rgba(200,230,255,0.9)")
                                 : (darkMode ? "rgba(255,100,100,0.3)" : "rgba(255,220,220,0.9)"));
              p.rect(px, py, step, step);
            }
          }
        }

        const weights = model.weights || [];
        const bias = model.bias || 0;

        // draw line for linear
        if (classifierType === "linear" && weights.length >= 2 && Math.abs(weights[1]) > 1e-6) {
          p.stroke(darkMode ? 255 : 0, 150, 255);
          const x1 = -1, x2 = 1;
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
          p.fill(pt.labelSigned === 1 ? (darkMode ? "cyan" : "blue") : (darkMode ? "orange" : "red"));
          p.stroke(0);
          p.circle(px, py, 10);
          if (predSigned !== pt.labelSigned) { p.noFill(); p.stroke(0); p.circle(px, py, 14); }
        }

        // hover-based equation display
        let showEquation = false;
        const mouseXNorm = p.map(p.mouseX, 0, p.width, -1, 1);
        const mouseYNorm = p.map(p.mouseY, p.height, 0, -1, 1);
        if (classifierType === "linear" && weights.length >= 2 && Math.abs(weights[1]) > 1e-6) {
          const yOnLine = -(weights[0]*mouseXNorm + bias)/weights[1];
          if (Math.abs(yOnLine - mouseYNorm) < 0.05) showEquation = true;
        } else if (classifierType === "poly") {
          if (Math.abs(safePredict([mouseXNorm, mouseYNorm]) - 0.5) < 0.1) showEquation = true;
        }

        if (equationRef.current) {
          equationRef.current.style.opacity = showEquation ? 1 : 0;
          if (showEquation) {
            equationRef.current.textContent =
              classifierType === "linear" ? buildLinearEquation(weights, bias)
                                           : buildPolyEquation(weights, bias);
          }
        }
      };

      // keyboard controls
      p.keyPressed = () => {
        const mouseXNorm = p.map(p.mouseX, 0, p.width, -1, 1);
        const mouseYNorm = p.map(p.mouseY, p.height, 0, -1, 1);
        if (p.key === " ") { paused = !paused; return; }
        if (p.key === "b") { points.push({ x: mouseXNorm, y: mouseYNorm, labelSigned: 1 }); X.push([mouseXNorm, mouseYNorm]); y.push(1); safeTrainSample([mouseXNorm, mouseYNorm], 1); }
        if (p.key === "r") { points.push({ x: mouseXNorm, y: mouseYNorm, labelSigned: -1 }); X.push([mouseXNorm, mouseYNorm]); y.push(0); safeTrainSample([mouseXNorm, mouseYNorm], 0); }
        if (p.key === "R") reset();
        if (p.key === "+" || p.key === "=") speed = Math.min(20, speed + 1);
        if (p.key === "-" || p.key === "_") speed = Math.max(1, speed - 1);
      };
    };

    if (p5InstanceRef.current) { p5InstanceRef.current.remove(); p5InstanceRef.current = null; }
    if (sketchRef.current) { p5InstanceRef.current = new p5(sketch, sketchRef.current); }

    return () => { if (p5InstanceRef.current) { p5InstanceRef.current.remove(); p5InstanceRef.current = null; } };
  }, [classifierType]);

  return (
    <div ref={sketchRef} style={{ position: "relative", width: "100vw", height: "calc(100vh - 60px)", overflow: "hidden" }}>
      <div ref={equationRef} style={{
        position: "absolute", top: "50%", left: "50%", transform: "translate(-50%, -50%)",
        background: "rgba(255,255,255,0.85)", padding: "6px 12px", borderRadius: 10,
        fontFamily: "monospace", fontSize: 16, zIndex: 100,
        boxShadow: "0 4px 14px rgba(0,0,0,0.15)", opacity: 0,
        transition: "opacity 0.3s ease", pointerEvents: "none"
      }} />
    </div>
  );
};

export default PerceptronDemo;
