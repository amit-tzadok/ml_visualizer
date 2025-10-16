import React, { useRef, useEffect, useState } from "react";
import p5 from "p5";
import { predictKNN } from "../utils/knn";

const KNNDemo = ({
  dataset: externalDataset,
  onDatasetChange,
  showInstructions = true,
}) => {
  const sketchRef = useRef();
  const p5InstanceRef = useRef();
  const [k, setK] = useState(3);
  const kRef = useRef(k);
  const pointsRef = useRef([]);
  const modeRef = useRef("addA"); // "addA", "addB", "predict"
  const lastPredictionRef = useRef(null);

  useEffect(() => {
    // keep kRef in sync with state so p5 callbacks can read the latest value
    kRef.current = k;

    const sketch = (p) => {
      const pointRadius = 6;
      let points = pointsRef.current;
      let mode = modeRef.current;

      p.setup = () => {
        if (sketchRef.current)
          sketchRef.current
            .querySelectorAll("canvas")
            .forEach((c) => c.remove());
        const rect = sketchRef.current?.getBoundingClientRect();
        const w = Math.max(300, rect?.width || 600);
        const h = Math.max(300, rect?.height || 600);
        p.createCanvas(w, h);
        p.background(255);
        p.textAlign(p.CENTER, p.CENTER);
        p.textSize(16);
        // initialize points from external dataset if provided (convert -1..1 to pixels)
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
          `Current mode: ${
            mode === "addA"
              ? "Add Class A"
              : mode === "addB"
              ? "Add Class B"
              : "Predict"
          }`,
          p.width / 2,
          100
        );
      };

      p.windowResized = () => {
        const rect = sketchRef.current?.getBoundingClientRect();
        const newW = Math.max(300, rect?.width || 600);
        const newH = Math.max(300, rect?.height || 600);
        p.resizeCanvas(newW, newH);
        // remap points from externalDataset if provided
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

        // draw base points
        pts.forEach((pt) => {
          const fillColor =
            pt.label === "A" ? "red" : pt.label === "B" ? "blue" : "green";
          p.fill(fillColor);
          p.noStroke();
          p.ellipse(pt.coords[0], pt.coords[1], pointRadius * 2);
        });

        // animated neighbor highlights + connecting lines
        const lp = lastPredictionRef.current;
        if (lp && Array.isArray(lp.neighbors) && lp.neighbors.length > 0) {
          const predictedCoords = lp.coords;
          lp.neighbors.forEach((nIdx, i) => {
            const npt = pts[nIdx];
            if (!npt) return;
            p.push();
            // semi-transparent fill color by class
            let col;
            if (npt.label === "A") col = p.color(255, 80, 80, 160);
            else if (npt.label === "B") col = p.color(80, 140, 255, 160);
            else col = p.color(80, 255, 120, 160);

            // static highlight (no pulsing) for better performance in compare mode
            const isCompare =
              externalDataset &&
              Array.isArray(externalDataset) &&
              externalDataset.length;
            const glowR = pointRadius * (isCompare ? 4 : 6);
            p.noStroke();
            p.fill(col);
            p.ellipse(npt.coords[0], npt.coords[1], glowR);

            // bold outline
            p.stroke(p.red(col), p.green(col), p.blue(col), 200);
            p.strokeWeight(isCompare ? 2 : 4);
            p.noFill();
            p.circle(npt.coords[0], npt.coords[1], glowR * 0.9);
            p.pop();

            // connecting line to predicted point
            if (predictedCoords) {
              p.stroke(60, 60, 60, 160);
              p.strokeWeight(2);
              p.line(
                predictedCoords[0],
                predictedCoords[1],
                npt.coords[0],
                npt.coords[1]
              );
            }
          });

          // draw predicted point marker larger
          if (lp.coords) {
            p.push();
            p.noStroke();
            p.fill(0, 200, 100, 220);
            p.ellipse(lp.coords[0], lp.coords[1], pointRadius * 4);
            p.pop();
          }
        }

        // draw predicted label text
        if (lp) {
          p.fill("black");
          p.noStroke();
          p.textSize(14);
          p.text(`Predicted: ${lp.label}`, p.width / 2, p.height - 18);
        }
      };

      p.mousePressed = () => {
        if (
          p.mouseX < 0 ||
          p.mouseX > p.width ||
          p.mouseY < 0 ||
          p.mouseY > p.height
        )
          return;
        if (mode === "addA") {
          points.push({ coords: [p.mouseX, p.mouseY], label: "A" });
          if (onDatasetChange) {
            // convert to normalized coords -1..1
            const nx = p.map(p.mouseX, 0, p.width, -1, 1);
            const ny = p.map(p.mouseY, p.height, 0, -1, 1);
            onDatasetChange((d) => [...d, { x: nx, y: ny, label: "A" }]);
          }
          // if there is an existing predicted point, recompute its prediction/neighbors
          if (lastPredictionRef.current && lastPredictionRef.current.coords) {
            const sample = lastPredictionRef.current.coords;
            const baseIdx = points
              .map((pt, i) => ({ pt, i }))
              .filter(({ pt }) => pt.label === "A" || pt.label === "B");
            const predictedLabelNow = predictKNN(
              baseIdx.map(({ pt }) => pt),
              kRef.current,
              sample
            );
            const dists = baseIdx.map(({ pt, i }) => ({
              i,
              dist: Math.hypot(
                pt.coords[0] - sample[0],
                pt.coords[1] - sample[1]
              ),
            }));
            dists.sort((a, b) => a.dist - b.dist);
            const neighborIndicesNow = dists
              .slice(0, Math.max(1, Math.min(kRef.current, dists.length)))
              .map((n) => n.i);
            lastPredictionRef.current = {
              label: predictedLabelNow,
              neighbors: neighborIndicesNow,
              coords: sample,
            };
          }
          drawPoints();
        } else if (mode === "addB") {
          points.push({ coords: [p.mouseX, p.mouseY], label: "B" });
          if (onDatasetChange) {
            const nx = p.map(p.mouseX, 0, p.width, -1, 1);
            const ny = p.map(p.mouseY, p.height, 0, -1, 1);
            onDatasetChange((d) => [...d, { x: nx, y: ny, label: "B" }]);
          }
          // recompute neighbors/prediction for existing predicted point
          if (lastPredictionRef.current && lastPredictionRef.current.coords) {
            const sample = lastPredictionRef.current.coords;
            const baseIdx = points
              .map((pt, i) => ({ pt, i }))
              .filter(({ pt }) => pt.label === "A" || pt.label === "B");
            const predictedLabelNow = predictKNN(
              baseIdx.map(({ pt }) => pt),
              kRef.current,
              sample
            );
            const dists = baseIdx.map(({ pt, i }) => ({
              i,
              dist: Math.hypot(
                pt.coords[0] - sample[0],
                pt.coords[1] - sample[1]
              ),
            }));
            dists.sort((a, b) => a.dist - b.dist);
            const neighborIndicesNow = dists
              .slice(0, Math.max(1, Math.min(kRef.current, dists.length)))
              .map((n) => n.i);
            lastPredictionRef.current = {
              label: predictedLabelNow,
              neighbors: neighborIndicesNow,
              coords: sample,
            };
          }
          drawPoints();
        } else if (mode === "predict") {
          if (
            points.filter((pt) => pt.label === "A" || pt.label === "B").length <
            kRef.current
          ) {
            alert(`Need at least ${k} points from each class to predict.`);
            return;
          }
          const baseIdx = points
            .map((pt, i) => ({ pt, i }))
            .filter(({ pt }) => pt.label === "A" || pt.label === "B");
          const predictedLabel = predictKNN(
            baseIdx.map(({ pt }) => pt),
            kRef.current,
            [p.mouseX, p.mouseY]
          );
          points.push({ coords: [p.mouseX, p.mouseY], label: predictedLabel });
          // compute neighbor indices (relative to points array)
          const dists = baseIdx.map(({ pt, i }) => ({
            i,
            dist: Math.hypot(pt.coords[0] - p.mouseX, pt.coords[1] - p.mouseY),
          }));
          dists.sort((a, b) => a.dist - b.dist);
          const neighborIndices = dists
            .slice(0, Math.max(1, Math.min(kRef.current, dists.length)))
            .map((n) => n.i);
          lastPredictionRef.current = {
            label: predictedLabel,
            neighbors: neighborIndices,
            coords: [p.mouseX, p.mouseY],
          };
          if (onDatasetChange) {
            const nx = p.map(p.mouseX, 0, p.width, -1, 1);
            const ny = p.map(p.mouseY, p.height, 0, -1, 1);
            onDatasetChange((d) => [
              ...d,
              { x: nx, y: ny, label: predictedLabel },
            ]);
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
        // animate highlights each frame
        drawPoints();
      };
    };

    // (re)create p5 instance so we can reinitialize points when externalDataset or showInstructions changes
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

  // keep kRef in sync when k changes and recompute neighbors for last prediction
  useEffect(() => {
    kRef.current = k;
    const lp = lastPredictionRef.current;
    if (lp && lp.coords) {
      const pts = pointsRef.current || [];
      const baseIdx = pts
        .map((pt, i) => ({ pt, i }))
        .filter(({ pt }) => pt.label === "A" || pt.label === "B");
      const dists = baseIdx.map(({ pt, i }) => ({
        i,
        dist: Math.hypot(
          pt.coords[0] - lp.coords[0],
          pt.coords[1] - lp.coords[1]
        ),
      }));
      dists.sort((a, b) => a.dist - b.dist);
      const neighborIndices = dists
        .slice(0, Math.max(1, Math.min(kRef.current, dists.length)))
        .map((n) => n.i);
      lastPredictionRef.current = { ...lp, neighbors: neighborIndices };
    }
  }, [k]);

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
      {/* canvas will be created into this container */}
      <div style={{ position: "absolute", left: 0, top: 0, right: 0 }} />
      <div style={{ marginTop: "10px", position: "relative", zIndex: 10 }}>
        <label>
          k (number of neighbors):
          <input
            type="number"
            value={k}
            min={1}
            max={20}
            onChange={(e) =>
              setK(Math.max(1, Math.min(20, parseInt(e.target.value) || 1)))
            }
            style={{ marginLeft: "10px", width: "50px" }}
          />
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
            <li>
              In prediction mode, click to classify a new point based on
              existing points.
            </li>
          </ul>
        </div>
      )}
    </div>
  );
};

export default KNNDemo;
