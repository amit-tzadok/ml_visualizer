import React, { useRef, useEffect, useState } from "react";
import p5 from "p5";
import { predictKNN } from "../utils/knn";

import type { Point as TPoint } from "../types";
import type { P5Instance } from "../types";

type KnnProps = {
  dataset?: TPoint[];
  onDatasetChange?: (updater: ((d: TPoint[]) => TPoint[]) | TPoint[]) => void;
  showInstructions?: boolean;
};

const KNNDemo: React.FC<KnnProps> = ({ dataset: externalDataset, onDatasetChange, showInstructions = true }) => {
  const sketchRef = useRef<HTMLDivElement | null>(null);
  const p5InstanceRef = useRef<any>(null);
  const [k, setK] = useState<number>(3);
  const kRef = useRef<number>(k);
  const pointsRef = useRef<any[]>([]);
  const modeRef = useRef<string>("addA");
  const lastPredictionRef = useRef<any>(null);

  useEffect(() => {
    kRef.current = k;

    const sketch = (p: any) => {
      const pointRadius = 6;
      let points = pointsRef.current;
      let mode = modeRef.current;

      p.setup = () => {
        if (sketchRef.current)
          sketchRef.current.querySelectorAll("canvas").forEach((c) => c.remove());
        const rect = sketchRef.current?.getBoundingClientRect();
        const w = Math.max(300, rect?.width || 600);
        const h = Math.max(300, rect?.height || 600);
        p.createCanvas(w, h);
        p.background(255);
        p.textAlign(p.CENTER, p.CENTER);
        p.textSize(16);
        points = pointsRef.current = [];
        if (externalDataset && Array.isArray(externalDataset)) {
          for (const pt of externalDataset) {
            const px = p.map(pt.x, -1, 1, 0, p.width);
            const py = p.map(pt.y, -1, 1, p.height, 0);
            points.push({ coords: [px, py], label: pt.label });
          }
        }
        drawInstructions();
      };

      const drawInstructions = () => {
        if (!showInstructions) return;
        p.fill(0);
        p.noStroke();
        p.text("Click to add points", p.width / 2, 20);
        p.text("Press A to add Class A (red)", p.width / 2, 40);
        p.text("Press B to add Class B (blue)", p.width / 2, 60);
        p.text("Press P to predict class (green)", p.width / 2, 80);
        p.text(
          `Current mode: ${mode === "addA" ? "Add Class A" : mode === "addB" ? "Add Class B" : "Predict"}`,
          p.width / 2,
          100
        );
      };

      p.windowResized = () => {
        const rect = sketchRef.current?.getBoundingClientRect();
        const newW = Math.max(300, rect?.width || 600);
        const newH = Math.max(300, rect?.height || 600);
        p.resizeCanvas(newW, newH);
        if (externalDataset && Array.isArray(externalDataset)) {
          points = pointsRef.current = [];
          for (const pt of externalDataset) {
            const px = p.map(pt.x, -1, 1, 0, p.width);
            const py = p.map(pt.y, -1, 1, p.height, 0);
            points.push({ coords: [px, py], label: pt.label });
          }
        }
      };

      const drawPoints = () => {
        const pts = pointsRef.current || [];
        p.clear();
        p.background(255);
        drawInstructions();

        pts.forEach((pt) => {
          const fillColor = pt.label === "A" ? "red" : pt.label === "B" ? "blue" : "green";
          p.fill(fillColor);
          p.noStroke();
          p.ellipse(pt.coords[0], pt.coords[1], pointRadius * 2);
        });

        const lp = lastPredictionRef.current;
        if (lp && Array.isArray(lp.neighbors) && lp.neighbors.length > 0) {
          const predictedCoords = lp.coords;
          lp.neighbors.forEach((nIdx: number, i: number) => {
            const npt = pts[nIdx];
            if (!npt) return;
            p.push();
            let col;
            if (npt.label === "A") col = p.color(255, 80, 80, 160);
            else if (npt.label === "B") col = p.color(80, 140, 255, 160);
            else col = p.color(80, 255, 120, 160);

            const isCompare = externalDataset && Array.isArray(externalDataset) && externalDataset.length;
            const glowR = pointRadius * (isCompare ? 4 : 6);
            p.noStroke();
            p.fill(col);
            p.ellipse(npt.coords[0], npt.coords[1], glowR);

            p.stroke(p.red(col), p.green(col), p.blue(col), 200);
            p.strokeWeight(isCompare ? 2 : 4);
            p.noFill();
            p.circle(npt.coords[0], npt.coords[1], glowR * 0.9);
            p.pop();

            if (predictedCoords) {
              p.stroke(60, 60, 60, 160);
              p.strokeWeight(2);
              p.line(predictedCoords[0], predictedCoords[1], npt.coords[0], npt.coords[1]);
            }
          });

          if (lp.coords) {
            p.push();
            p.noStroke();
            p.fill(0, 200, 100, 220);
            p.ellipse(lp.coords[0], lp.coords[1], pointRadius * 4);
            p.pop();
          }
        }

        if (lp) {
          p.fill("black");
          p.noStroke();
          p.textSize(14);
          p.text(`Predicted: ${lp.label}`, p.width / 2, p.height - 18);
        }
      };

      p.mousePressed = () => {
        if (p.mouseX < 0 || p.mouseX > p.width || p.mouseY < 0 || p.mouseY > p.height) return;
        if (mode === "addA") {
          points.push({ coords: [p.mouseX, p.mouseY], label: "A" });
          if (onDatasetChange) {
            const nx = p.map(p.mouseX, 0, p.width, -1, 1);
            const ny = p.map(p.mouseY, p.height, 0, -1, 1);
            onDatasetChange((d) => [...(Array.isArray(d) ? d : []), { x: nx, y: ny, label: "A" }]);
          }
          if (lastPredictionRef.current && lastPredictionRef.current.coords) {
            const sample = lastPredictionRef.current.coords;
            const baseIdx = points
              .map((pt, i) => ({ pt, i }))
              .filter(({ pt }) => pt.label === "A" || pt.label === "B");
            const predictedLabelNow = predictKNN(baseIdx.map(({ pt }) => pt), kRef.current, sample);
            const dists = baseIdx.map(({ pt, i }) => ({ i, dist: Math.hypot(pt.coords[0] - sample[0], pt.coords[1] - sample[1]) }));
            dists.sort((a, b) => a.dist - b.dist);
            const neighborIndicesNow = dists.slice(0, Math.max(1, Math.min(kRef.current, dists.length))).map((n) => n.i);
            lastPredictionRef.current = { label: predictedLabelNow, neighbors: neighborIndicesNow, coords: sample };
          }
          drawPoints();
        } else if (mode === "addB") {
          points.push({ coords: [p.mouseX, p.mouseY], label: "B" });
          if (onDatasetChange) {
            const nx = p.map(p.mouseX, 0, p.width, -1, 1);
            const ny = p.map(p.mouseY, p.height, 0, -1, 1);
            onDatasetChange((d) => [...(Array.isArray(d) ? d : []), { x: nx, y: ny, label: "B" }]);
          }
          if (lastPredictionRef.current && lastPredictionRef.current.coords) {
            const sample = lastPredictionRef.current.coords;
            const baseIdx = points
              .map((pt, i) => ({ pt, i }))
              .filter(({ pt }) => pt.label === "A" || pt.label === "B");
            const predictedLabelNow = predictKNN(baseIdx.map(({ pt }) => pt), kRef.current, sample);
            const dists = baseIdx.map(({ pt, i }) => ({ i, dist: Math.hypot(pt.coords[0] - sample[0], pt.coords[1] - sample[1]) }));
            dists.sort((a, b) => a.dist - b.dist);
            const neighborIndicesNow = dists.slice(0, Math.max(1, Math.min(kRef.current, dists.length))).map((n) => n.i);
            lastPredictionRef.current = { label: predictedLabelNow, neighbors: neighborIndicesNow, coords: sample };
          }
          drawPoints();
        } else if (mode === "predict") {
          if (points.filter((pt) => pt.label === "A" || pt.label === "B").length < kRef.current) {
            alert(`Need at least ${k} points from each class to predict.`);
            return;
          }
          const baseIdx = points
            .map((pt, i) => ({ pt, i }))
            .filter(({ pt }) => pt.label === "A" || pt.label === "B");
          const predictedLabel = predictKNN(baseIdx.map(({ pt }) => pt), kRef.current, [p.mouseX, p.mouseY]);
          points.push({ coords: [p.mouseX, p.mouseY], label: predictedLabel });
          const dists = baseIdx.map(({ pt, i }) => ({ i, dist: Math.hypot(pt.coords[0] - p.mouseX, pt.coords[1] - p.mouseY) }));
          dists.sort((a, b) => a.dist - b.dist);
          const neighborIndices = dists.slice(0, Math.max(1, Math.min(kRef.current, dists.length))).map((n) => n.i);
          lastPredictionRef.current = { label: predictedLabel, neighbors: neighborIndices, coords: [p.mouseX, p.mouseY] };
          if (onDatasetChange) {
            const nx = p.map(p.mouseX, 0, p.width, -1, 1);
            const ny = p.map(p.mouseY, p.height, 0, -1, 1);
            onDatasetChange((d) => [...(Array.isArray(d) ? d : []), { x: nx, y: ny, label: predictedLabel }]);
          }
          drawPoints();
        }
      };

      p.keyPressed = () => {
        if (p.key === "a" || p.key === "A") {
          mode = "addA";
        } else if (p.key === "b" || p.key === "B") {
          mode = "addB";
        } else if (p.key === "p" || p.key === "P") {
          mode = "predict";
        }
        modeRef.current = mode;
        drawPoints();
      };

      p.draw = () => {
        drawPoints();
      };

      p.resetDemo = () => {
        pointsRef.current = [];
        lastPredictionRef.current = null;
        try {
          if (onDatasetChange) onDatasetChange([]);
        } catch {}
        drawPoints();
      };
    };

    if (p5InstanceRef.current) {
      p5InstanceRef.current.remove();
      p5InstanceRef.current = null;
    }
    p5InstanceRef.current = new p5(sketch, sketchRef.current);

    return () => {
      if (p5InstanceRef.current) {
        p5InstanceRef.current.remove();
        p5InstanceRef.current = null;
      }
    };
  }, [externalDataset, showInstructions]);

  useEffect(() => {
    kRef.current = k;
    const lp = lastPredictionRef.current;
    if (lp && lp.coords) {
      const pts = pointsRef.current || [];
      const baseIdx = pts
        .map((pt, i) => ({ pt, i }))
        .filter(({ pt }) => pt.label === "A" || pt.label === "B");
      const dists = baseIdx.map(({ pt, i }) => ({ i, dist: Math.hypot(pt.coords[0] - lp.coords[0], pt.coords[1] - lp.coords[1]) }));
      dists.sort((a, b) => a.dist - b.dist);
      const neighborIndices = dists.slice(0, Math.max(1, Math.min(kRef.current, dists.length))).map((n) => n.i);
      lastPredictionRef.current = { ...lp, neighbors: neighborIndices };
    }
  }, [k]);

  return (
    <div ref={sketchRef} style={{ position: "relative", width: "100%", height: "100%", overflow: "hidden" }}>
      <div style={{ position: "absolute", left: 0, top: 0, right: 0 }} />
      <div style={{ marginTop: "10px", position: "relative", zIndex: 10 }}>
        <button
          id="knn-reset"
          onClick={() => {
            if (p5InstanceRef.current && typeof p5InstanceRef.current.resetDemo === "function") {
              p5InstanceRef.current.resetDemo();
            }
          }}
          style={{ marginRight: "10px", padding: "4px 8px", borderRadius: 8 }}
        >
          Reset
        </button>
        <label>
          k (number of neighbors):
          <input type="number" value={k} min={1} max={20} onChange={(e) => setK(Math.max(1, Math.min(20, parseInt(e.target.value) || 1)))} style={{ marginLeft: "10px", width: "50px" }} />
        </label>
      </div>
      {showInstructions && (
        <div style={{ marginTop: "10px" }}>
          <strong>Instructions:</strong>
          <ul>
            <li>Click on the canvas to add points.</li>
            <li>Press 'A' to switch to adding Class A points (red).</li>
            <li>Press 'B' to switch to adding Class B points (blue).</li>
            <li>Press 'P' to switch to prediction mode (green).</li>
            <li>In prediction mode, click to classify a new point based on existing points.</li>
          </ul>
        </div>
      )}
    </div>
  );
};

export default KNNDemo;
