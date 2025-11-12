import React, { useRef, useEffect, useState } from "react";
import { predictKNN } from "../utils/knn";
import loadP5 from "../utils/loadP5";

import type { Point as TPoint, Predicted } from "../types";
import type { P5Instance } from "../types/p5";

type KnnProps = {
  dataset?: TPoint[];
  onDatasetChange?: (updater: ((d: TPoint[]) => TPoint[]) | TPoint[]) => void;
  showInstructions?: boolean;
};

const KNNDemo: React.FC<KnnProps> = ({
  dataset: externalDataset,
  onDatasetChange,
  showInstructions = true,
}) => {
  const sketchRef = useRef<HTMLDivElement | null>(null);
  const p5InstanceRef = useRef<P5Instance | null>(null);
  const [loading, setLoading] = useState(true);
  const [k, setK] = useState<number>(3);
  const kRef = useRef<number>(k);
  const pointsRef = useRef<
    Array<{ coords: [number, number]; label: string | number }>
  >([]);
  const modeRef = useRef<string>("addA");
  const lastPredictionRef = useRef<Predicted | null>(null);
  const onDatasetChangeRef = useRef<typeof onDatasetChange | null>(
    onDatasetChange
  );

  // Theme detection
  const prefersDark =
    typeof window !== "undefined" &&
    window.matchMedia &&
    window.matchMedia("(prefers-color-scheme: dark)").matches;
  const panelBg = prefersDark
    ? "rgba(45,55,72,0.95)"
    : "rgba(255,255,255,0.95)";
  const panelText = prefersDark ? "#f7fafc" : "#1a202c";
  const panelBorder = prefersDark
    ? "1px solid rgba(255,255,255,0.1)"
    : "1px solid rgba(0,0,0,0.1)";
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
    // keep onDatasetChange ref current for use inside the p5 sketch (avoid recreating sketch)
    const sketch = (p: P5Instance) => {
      type P5KnnExt = {
        recomputeForK?: (n?: number) => void;
        requestRedraw?: () => void;
        [k: string]: unknown;
      };
      const pEx = p as unknown as P5Instance & P5KnnExt;
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
        // Darken slightly (250 -> 240) for consistency with updated demo backgrounds
        p.background(240);
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
        // initial paint
        needsRedraw = true;
      };

      const drawInstructions = () => {
        // Instructions moved to UI overlay
      };

      let needsRedraw = true;
      const requestRedraw = () => {
        needsRedraw = true;
      };

      // Expose a method to recompute neighbor highlights for current K and redraw
      const recomputeForK = (nextK?: number) => {
        try {
          const lp = lastPredictionRef.current;
          if (!lp || !lp.coords) {
            requestRedraw();
            return;
          }
          const baseIdx = points
            .map((pt, i: number) => ({ pt, i }))
            .filter(({ pt }) => pt.label === "A" || pt.label === "B");
          const kUse = Math.max(
            1,
            Math.min(nextK ?? kRef.current, baseIdx.length)
          );
          const dists = baseIdx.map(({ pt, i }) => ({
            i,
            dist: Math.hypot(
              pt.coords[0] - lp.coords[0],
              pt.coords[1] - lp.coords[1]
            ),
          }));
          dists.sort((a, b) => a.dist - b.dist);
          const neighborIndices = dists.slice(0, kUse).map((n) => n.i);
          lastPredictionRef.current = { ...lp, neighbors: neighborIndices };
          requestRedraw();
        } catch (err) {
          requestRedraw();
          if (isDev()) console.debug("knn: recomputeForK error", err);
        }
      };

      // attach helpers on the p5 instance (augment at runtime)
      try {
        pEx.recomputeForK = recomputeForK;
        pEx.requestRedraw = requestRedraw;
      } catch (err) {
        // ignore attach errors but surface in dev
        if (isDev()) console.debug("knn: attach helper error", err);
      }

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
        requestRedraw();
      };

      const drawPoints = () => {
        const pts = points || [];
        p.clear();
        // Darken slightly (250 -> 240) for consistency with updated demo backgrounds
        p.background(240);
        drawInstructions();

        pts.forEach((pt) => {
          const fillColor =
            pt.label === "A" ? "red" : pt.label === "B" ? "blue" : "green";
          p.fill(fillColor);
          p.noStroke();
          p.ellipse(pt.coords[0], pt.coords[1], pointRadius * 2);
        });

        const lp = lastPredictionRef.current;
        if (lp && Array.isArray(lp.neighbors) && lp.neighbors.length > 0) {
          const predictedCoords = lp.coords;
          lp.neighbors.forEach((nIdx: number) => {
            const npt = pts[nIdx];
            if (!npt) return;
            p.push();
            let col;
            if (npt.label === "A") col = p.color(255, 80, 80, 160);
            else if (npt.label === "B") col = p.color(80, 140, 255, 160);
            else col = p.color(80, 255, 120, 160);

            const isCompare =
              externalDataset &&
              Array.isArray(externalDataset) &&
              externalDataset.length;
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
              p.line(
                predictedCoords[0],
                predictedCoords[1],
                npt.coords[0],
                npt.coords[1]
              );
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
        needsRedraw = false;
      };

      p.mousePressed = () => {
        if (
          p.mouseX < 0 ||
          p.mouseX > p.width ||
          p.mouseY < 0 ||
          p.mouseY > p.height
        )
          return;
        const controlled = typeof onDatasetChangeRef.current === "function";
        if (mode === "addA") {
          if (controlled) {
            // In controlled mode (compare view), update parent dataset only
            const nx = p.map(p.mouseX, 0, p.width, -1, 1);
            const ny = p.map(p.mouseY, p.height, 0, -1, 1);
            onDatasetChangeRef.current?.((d: TPoint[]) => [
              ...(Array.isArray(d) ? d : []),
              { x: nx, y: ny, label: "A" },
            ]);
            return; // parent re-renders and sketch rebuilds from dataset
          }
          // Uncontrolled: update local points
          points.push({ coords: [p.mouseX, p.mouseY], label: "A" });
          if (onDatasetChangeRef.current) {
            const nx = p.map(p.mouseX, 0, p.width, -1, 1);
            const ny = p.map(p.mouseY, p.height, 0, -1, 1);
            onDatasetChangeRef.current((d: TPoint[]) => [
              ...(Array.isArray(d) ? d : []),
              { x: nx, y: ny, label: "A" },
            ]);
          }
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
          requestRedraw();
        } else if (mode === "addB") {
          if (controlled) {
            const nx = p.map(p.mouseX, 0, p.width, -1, 1);
            const ny = p.map(p.mouseY, p.height, 0, -1, 1);
            onDatasetChangeRef.current?.((d: TPoint[]) => [
              ...(Array.isArray(d) ? d : []),
              { x: nx, y: ny, label: "B" },
            ]);
            return;
          }
          points.push({ coords: [p.mouseX, p.mouseY], label: "B" });
          if (onDatasetChangeRef.current) {
            const nx = p.map(p.mouseX, 0, p.width, -1, 1);
            const ny = p.map(p.mouseY, p.height, 0, -1, 1);
            onDatasetChangeRef.current((d: TPoint[]) => [
              ...(Array.isArray(d) ? d : []),
              { x: nx, y: ny, label: "B" },
            ]);
          }
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
          requestRedraw();
        } else if (mode === "predict") {
          if (
            points.filter((pt) => pt.label === "A" || pt.label === "B").length <
            kRef.current
          ) {
            // Not enough points to make a prediction; silently ignore
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
          if (controlled) {
            const nx = p.map(p.mouseX, 0, p.width, -1, 1);
            const ny = p.map(p.mouseY, p.height, 0, -1, 1);
            onDatasetChangeRef.current?.((d: TPoint[]) => [
              ...(Array.isArray(d) ? d : []),
              { x: nx, y: ny, label: predictedLabel },
            ]);
          } else {
            points.push({
              coords: [p.mouseX, p.mouseY],
              label: predictedLabel,
            });
          }
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
          requestRedraw();
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
        requestRedraw();
      };

      p.draw = () => {
        if (needsRedraw) drawPoints();
      };

      p.resetDemo = () => {
        points.length = 0; // Clear array in place
        lastPredictionRef.current = null;
        try {
          if (onDatasetChangeRef.current) onDatasetChangeRef.current([]);
        } catch (err) {
          // Surface dataset callback problems during development but keep production quiet
          if (isDev()) console.debug("knn: onDatasetChange error", err);
        }
        requestRedraw();
      };
    };

    let mounted = true;
    setLoading(true);
    (async () => {
      try {
        const mod = await loadP5();
        if (!mounted) return;
        // `mod` may be a module namespace with a default export or the constructor itself.
        // Narrow the cast without using `any` to satisfy eslint's no-explicit-any rule.
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const P5 = (mod && (mod as any).default) || mod;
        if (p5InstanceRef.current) {
          p5InstanceRef.current.remove();
          p5InstanceRef.current = null;
        }
        p5InstanceRef.current = new P5(
          sketch,
          sketchRef.current
        ) as unknown as P5Instance;
        setLoading(false);
      } catch (err) {
        if (isDev()) console.debug("knn: dynamic import p5 failed", err);
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
  }, [externalDataset, showInstructions]);

  // keep onDatasetChangeRef in sync with prop changes
  useEffect(() => {
    onDatasetChangeRef.current = onDatasetChange;
  }, [onDatasetChange]);

  // update kRef and ask p5 to recompute when K changes
  useEffect(() => {
    kRef.current = k;
    try {
      if (
        p5InstanceRef.current &&
        typeof p5InstanceRef.current.recomputeForK === "function"
      ) {
        p5InstanceRef.current.recomputeForK(k);
      }
    } catch (err) {
      if (isDev()) console.debug("knn: recompute trigger error", err);
    }
  }, [k]);

  return (
    <div
      ref={sketchRef}
      style={{
        position: "relative",
        width: "calc(100% - 280px)",
        height: "92%",
        overflow: "hidden",
        marginLeft: "-200px",
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
      {/* Clear button */}
      <div
        style={{
          position: "absolute",
          left: 12,
          top: 12,
          zIndex: 20,
        }}
      >
        <button
          id="knn-clear"
          onClick={() => {
            if (
              p5InstanceRef.current &&
              typeof p5InstanceRef.current.resetDemo === "function"
            ) {
              p5InstanceRef.current.resetDemo();
            }
          }}
          style={{
            padding: "8px 16px",
            fontSize: 12,
            fontWeight: 600,
            background: panelBg,
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
          🧹 Clear
        </button>
      </div>

      {/* Instructions overlay */}
      {showInstructions && (
        <div
          style={{
            position: "absolute",
            left: 12,
            top: 60,
            zIndex: 20,
            maxWidth: 280,
            background: panelBg,
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
              🔍 K-Nearest Neighbors
            </div>
            <div style={{ fontSize: 11, opacity: 0.9 }}>
              <span style={{ fontWeight: 600 }}>Click:</span> Add points
              <br />
              <span style={{ fontWeight: 600 }}>A key:</span> Red (Class A)
              <br />
              <span style={{ fontWeight: 600 }}>B key:</span> Blue (Class B)
              <br />
              <span style={{ fontWeight: 600 }}>P key:</span> Predict (Green)
            </div>
          </div>
        </div>
      )}

      {/* Control panel - K value */}
      <div
        style={{
          position: "fixed",
          right: 16,
          top: 86,
          zIndex: 1150,
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

        <div style={{ marginBottom: 6 }}>
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
            Parameters
          </div>
          <div>
            <label
              style={{
                fontSize: 11,
                color: panelText,
                opacity: 0.8,
                display: "block",
                marginBottom: 4,
              }}
            >
              K (Neighbors)
            </label>
            <input
              type="number"
              value={k}
              min={1}
              max={20}
              onChange={(e) =>
                setK(Math.max(1, Math.min(20, parseInt(e.target.value) || 1)))
              }
              onInput={(e) =>
                setK(
                  Math.max(
                    1,
                    Math.min(
                      20,
                      parseInt((e.target as HTMLInputElement).value) || 1
                    )
                  )
                )
              }
              style={{
                width: "100%",
                padding: "6px 8px",
                fontSize: 13,
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
            marginTop: 12,
            padding: "8px",
            background: prefersDark
              ? "rgba(255,255,255,0.05)"
              : "rgba(0,0,0,0.03)",
            borderRadius: 6,
            fontSize: 11,
            lineHeight: 1.5,
          }}
        >
          <div style={{ fontWeight: 600, marginBottom: 4, opacity: 0.7 }}>
            Current Mode:
          </div>
          <div
            style={{
              fontSize: 12,
              fontWeight: 600,
              color:
                modeRef.current === "addA"
                  ? "#e53e3e"
                  : modeRef.current === "addB"
                  ? "#3182ce"
                  : "#38a169",
            }}
          >
            {modeRef.current === "addA"
              ? "🔴 Add Class A (Red)"
              : modeRef.current === "addB"
              ? "🔵 Add Class B (Blue)"
              : "🟢 Predict Class"}
          </div>
        </div>
      </div>
    </div>
  );
};

export default KNNDemo;
