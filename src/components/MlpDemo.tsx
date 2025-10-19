import React, { useEffect, useRef } from "react";
import p5 from "p5";
import type { P5Instance } from "../types/p5";
import MLPClassifier from "../utils/mlp";

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

  useEffect(() => {
  const sketch = (p: P5Instance) => {
      let points: Point[] = [];
      let X: number[][] = [];
      let y: number[] = [];
      let model: MLPClassifier | null = null;

      const resetModel = (hidden: number[] = [16]) => {
        model = new MLPClassifier(2, hidden, { lr: 0.05 });
      };

      const resetDemo = () => {
        points = [];
        X = [];
        y = [];
        for (let i = 0; i < 30; i++) {
          const vx = p.random(-1, 1);
          const vy = p.random(-1, 1);
          const label = vx * 0.4 + 0.1 > vy ? "A" : "B";
          points.push({ x: vx, y: vy, label });
          X.push([vx, vy]);
          y.push(label === "A" ? 1 : 0);
        }
        resetModel();
      };

      p.setup = () => {
        // remove any existing canvases in the container
        if (sketchRef.current) {
          sketchRef.current.querySelectorAll("canvas").forEach((c) => c.remove());
        }
        const rect = sketchRef.current?.getBoundingClientRect();
        p.createCanvas(Math.max(300, rect?.width || 600), Math.max(300, rect?.height || 600));
        resetDemo();
        if (externalDataset && Array.isArray(externalDataset) && externalDataset.length) p.updateDataset(externalDataset);
      };

      p.draw = () => {
        p.background(255);
        // draw decision field (coarse grid)
        const step = 18;
        for (let gx = 0; gx < p.width; gx += step) {
          for (let gy = 0; gy < p.height; gy += step) {
            if (model && X.length) {
              const nx = p.map(gx + 0.5 * step, 0, p.width, -1, 1);
              const ny = p.map(gy + 0.5 * step, p.height, 0, -1, 1);
              const score = model.forward([nx, ny]);
              const c = p.lerpColor(p.color("#ffcccc"), p.color("#cce5ff"), score);
              p.noStroke();
              p.fill(c);
              p.rect(gx, gy, step, step);
            }
          }
        }

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
        const label = p.mouseButton === p.LEFT ? "A" : "B";
        points.push({ x: mx, y: my, label });
        X.push([mx, my]);
        y.push(label === "A" ? 1 : 0);
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
      };

      p.trainMLP = (opts: { epochs?: number } = {}) => {
        if (!model) resetModel();
        if (!X.length || !model) return;
        const epochs = opts.epochs || Math.max(5, Math.floor(20 / Math.max(1, speedScale)));
        model.fit(X, y, { epochs, lr: 0.05, batchSize: 8 });
      };

      p.resetDemo = () => {
        resetDemo();
        if (onDatasetChange) onDatasetChange([]);
      };
    };

    // teardown previous instance if any
    if (p5Ref.current) {
      p5Ref.current.remove();
      p5Ref.current = null;
    }
    p5Ref.current = new p5(sketch, sketchRef.current as Element);

    return () => {
      if (p5Ref.current) {
        p5Ref.current.remove();
        p5Ref.current = null;
      }
    };
  }, [externalDataset, onDatasetChange, speedScale]);

  // when external dataset prop changes, forward to sketch
  useEffect(() => {
    if (p5Ref.current && typeof p5Ref.current.updateDataset === "function") {
      p5Ref.current.updateDataset(externalDataset || []);
    }
  }, [externalDataset]);

  // trigger training when resetToken changes (used to re-run training externally)
  useEffect(() => {
    if (p5Ref.current && typeof p5Ref.current.trainMLP === "function") {
      // small timeout so UI updates can settle
      setTimeout(() => p5Ref.current.trainMLP({ epochs: Math.max(8, Math.floor(20 / Math.max(1, speedScale))) }), 50);
    }
  }, [resetToken, speedScale]);

  return (
    <div ref={sketchRef} style={{ position: "relative", width: "100%", height: "100%", overflow: "hidden" }}>
      {/* simple floating buttons that call into the sketch instance */}
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
    </div>
  );
};

export default MlpDemo;
