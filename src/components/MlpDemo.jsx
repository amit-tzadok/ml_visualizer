import React, { useEffect, useRef } from "react";
import p5 from "p5";
import MLPClassifier from "../utils/mlp";

const MlpDemo = ({
  dataset: externalDataset,
  onDatasetChange,
  resetToken,
  showInstructions = true,
}) => {
  const sketchRef = useRef();
  const p5InstanceRef = useRef();

  useEffect(() => {
    const sketch = (p) => {
      let points = [],
        X = [],
        y = [];
      let gridG = null;
      let model = null;
      let gridStep = 6;
      const defaultGridStep = 6;

      const resetModel = () => {
        model = new MLPClassifier(2, [16], { lr: 0.05 });
      };

      const reset = () => {
        // initialize random points similar to Perceptron demo
        points = [];
        X = [];
        y = [];
        for (let i = 0; i < 50; i++) {
          const vx = p.random(-1, 1);
          const vy = p.random(-1, 1);
          const label = vx * 0.5 + 0.2 > vy ? "A" : "B";
          points.push({ x: vx, y: vy, label });
          X.push([vx, vy]);
          y.push(label === "A" ? 1 : 0);
        }
        resetModel();
      };

      const updateFromDataset = (d) => {
        points = [];
        X = [];
        y = [];
        if (d && Array.isArray(d) && d.length) {
          for (const pt of d) {
            points.push({ x: pt.x, y: pt.y, label: pt.label });
            X.push([pt.x, pt.y]);
            y.push(pt.label === "A" ? 1 : 0);
          }
        }
      };

      p.updateDataset = (newDataset) => {
        updateFromDataset(newDataset || []);
        resetModel();
      };

      p.setup = () => {
        if (sketchRef.current)
          sketchRef.current
            .querySelectorAll("canvas")
            .forEach((c) => c.remove());
        const rect = sketchRef.current?.getBoundingClientRect();
        const w = Math.max(300, rect?.width || 600);
        const h = Math.max(300, rect?.height || 600);
        p.createCanvas(w, h);
        gridG = p.createGraphics(p.width, p.height);
        if (
          externalDataset &&
          Array.isArray(externalDataset) &&
          externalDataset.length
        ) {
          p.updateDataset(externalDataset);
          // auto-start training when an external dataset is provided
          if (typeof p.trainMLP === "function") p.trainMLP();
        } else {
          reset();
          // auto-start training on the default random dataset
          if (typeof p.trainMLP === "function") p.trainMLP();
        }
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
        // background
        const darkMode =
          window.matchMedia &&
          window.matchMedia("(prefers-color-scheme: dark)").matches;
        p.background(darkMode ? 30 : 255);

        // draw grid
        if (!gridG) gridG = p.createGraphics(p.width, p.height);
        gridG.clear();
        for (let px = 0; px < p.width; px += gridStep) {
          for (let py = 0; py < p.height; py += gridStep) {
            const vx = p.map(px + gridStep / 2, 0, p.width, -1, 1);
            const vy = p.map(py + gridStep / 2, p.height, 0, -1, 1);
            const prob = model ? model.forward([vx, vy]) : 0.5;
            if (prob >= 0.5) {
              gridG.noStroke();
              gridG.fill(
                darkMode ? "rgba(80,150,255,0.3)" : "rgba(200,230,255,0.9)"
              );
              gridG.rect(px, py, gridStep, gridStep);
            } else {
              gridG.noStroke();
              gridG.fill(
                darkMode ? "rgba(255,100,100,0.3)" : "rgba(255,220,220,0.9)"
              );
              gridG.rect(px, py, gridStep, gridStep);
            }
          }
        }
        p.image(gridG, 0, 0);

        // draw points
        for (const pt of points) {
          const px = p.map(pt.x, -1, 1, 0, p.width);
          const py = p.map(pt.y, -1, 1, p.height, 0);
          p.fill(
            pt.label === "A"
              ? darkMode
                ? "cyan"
                : "blue"
              : darkMode
              ? "orange"
              : "red"
          );
          p.stroke(0);
          p.circle(px, py, 10);
        }

        // update status overlay
        const countEl = sketchRef.current?.querySelector("#mlp-point-count");
        const statusEl = sketchRef.current?.querySelector("#mlp-status");
        if (countEl) countEl.textContent = String(points.length);
        if (statusEl) statusEl.textContent = isTraining ? "training" : "idle";
      };

      p.mousePressed = () => {
        const mx = p.map(p.mouseX, 0, p.width, -1, 1);
        const my = p.map(p.mouseY, p.height, 0, -1, 1);
        // left click add A, right click add B
        const label = p.mouseButton === p.LEFT ? "A" : "B";
        points.push({ x: mx, y: my, label });
        X.push([mx, my]);
        y.push(label === "A" ? 1 : 0);
        if (onDatasetChange)
          onDatasetChange((d) => [...d, { x: mx, y: my, label }]);
      };

      // training state for UI/debugging
      let isTraining = false;

      // expose train helper (runs per-epoch asynchronously so UI stays responsive)
      p.trainMLP = async (options = {}) => {
        if (!model) resetModel();
        if (!X.length) return;
        const opts = Object.assign(
          { epochs: 20, lr: 0.05, batchSize: 8, shuffle: true },
          options
        );
        model.lr = opts.lr;
        isTraining = true;
        // run one epoch at a time to keep UI responsive and allow redraws
        for (let e = 0; e < opts.epochs; e++) {
          model.fit(X, y, {
            epochs: 1,
            lr: opts.lr,
            batchSize: opts.batchSize,
            shuffle: opts.shuffle,
          });
          // allow the event loop to breathe so p5 can redraw and UI updates
          // eslint-disable-next-line no-await-in-loop
          await new Promise((r) => setTimeout(r, 0));
        }
        isTraining = false;
      };
    };

    if (p5InstanceRef.current) {
      p5InstanceRef.current.remove();
      p5InstanceRef.current = null;
    }
    if (sketchRef.current)
      p5InstanceRef.current = new p5(sketch, sketchRef.current);

    return () => {
      if (p5InstanceRef.current) {
        p5InstanceRef.current.remove();
        p5InstanceRef.current = null;
      }
    };
  }, [externalDataset, onDatasetChange]);

  useEffect(() => {
    if (
      p5InstanceRef.current &&
      typeof p5InstanceRef.current.updateDataset === "function"
    ) {
      p5InstanceRef.current.updateDataset(externalDataset || []);
      if (typeof p5InstanceRef.current.trainMLP === "function")
        p5InstanceRef.current.trainMLP();
    }
  }, [externalDataset]);

  useEffect(() => {
    if (
      p5InstanceRef.current &&
      typeof p5InstanceRef.current.trainMLP === "function"
    ) {
      // auto-train on reset token change
      p5InstanceRef.current.trainMLP();
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
      <div
        style={{
          position: "absolute",
          left: 12,
          top: 12,
          zIndex: 200,
          background: "rgba(255,255,255,0.85)",
          padding: "6px 8px",
          borderRadius: 8,
        }}
      >
        <div style={{ fontSize: 12 }}>
          Points: <strong id="mlp-point-count">—</strong>
        </div>
        <div style={{ fontSize: 12 }}>
          Status: <strong id="mlp-status">idle</strong>
        </div>
      </div>
    </div>
  );
};

export default MlpDemo;
