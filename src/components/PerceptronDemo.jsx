import React, { useEffect, useRef } from "react";
import p5 from "p5";
import Perceptron from "../utils/perceptron";

const PerceptronDemo = () => {
  const sketchRef = useRef();
  const p5InstanceRef = useRef(null);

  useEffect(() => {
    const sketch = (p) => {
      let points = [];
      // dataset arrays used to train the Perceptron implementation
      let X = [];
      let y = [];

      // create a perceptron that will run 1 epoch per call to fit()
      const model = new Perceptron(2, 0.05, 1);

      // controls for live training
      let paused = false;
      let speed = 1; // number of samples to train per frame
      let sampleIndex = 0;

      const reset = () => {
        points = [];
        X = [];
        y = [];
        sampleIndex = 0;
        // re-seed model weights/bias
        model.weights = Array.from({ length: 2 }, () => Math.random() - 0.5);
        model.bias = Math.random() - 0.5;
        for (let i = 0; i < 50; i++) {
          const vx = p.random(-1, 1);
          const vy = p.random(-1, 1);
          const labelSigned = vx * 0.5 + 0.2 > vy ? 1 : -1;
          const label = labelSigned === 1 ? 1 : 0;
          points.push({ x: vx, y: vy, labelSigned });
          X.push([vx, vy]);
          y.push(label);
        }
      };

      p.setup = () => {
        // Remove any existing canvas with the fixed id to ensure only one visualization
        const existing = document.getElementById('perceptron-canvas');
        if (existing) existing.remove();
        const renderer = p.createCanvas(400, 400);
        // renderer.elt is the underlying canvas element
        if (renderer && renderer.elt) renderer.elt.id = 'perceptron-canvas';
        reset();
      };

      p.draw = () => {
        if (!paused && X.length > 0) {
          // train `speed` samples per frame (wrap around dataset)
          for (let s = 0; s < speed; s++) {
            const idx = sampleIndex % X.length;
            const xi = X[idx];
            const yi = y[idx];
            model.trainSample(xi, yi);
            sampleIndex++;
          }
        }

        p.background(255);

        // Draw dividing line (model prediction) using model weights & bias
        const weights = model.weights;
        const bias = model.bias;

        p.stroke(0, 150, 255);
        if (weights.length >= 2 && Math.abs(weights[1]) > 1e-6) {
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

        // Draw points and indicate whether current model classifies them correctly
          for (let i = 0; i < points.length; i++) {
          const pt = points[i];
          p.stroke(0);
          const px = p.map(pt.x, -1, 1, 0, p.width);
          const py = p.map(pt.y, -1, 1, p.height, 0);

          let pred = 0;
          try {
            pred = model.predict([pt.x, pt.y]);
          } catch (e) {
            // if prediction fails (e.g. wrong input size) mark as 0
            pred = 0;
          }
          const predSigned = pred === 1 ? 1 : -1;

          p.fill(pt.labelSigned === 1 ? "blue" : "red");
          p.circle(px, py, 10);

          if (predSigned !== pt.labelSigned) {
            p.noFill();
            p.stroke(0);
            p.circle(px, py, 14);
          }
        }

        // draw simple UI text
        p.noStroke();
        p.fill(0);
        p.textSize(12);
        p.text(`paused: ${paused}  speed: ${speed}  samples trained: ${sampleIndex}`, 10, p.height - 10);
      };

      // keyboard controls
      p.keyPressed = () => {
        // toggle pause on space
        if (p.key === " ") {
          paused = !paused;
          return;
        }

        // Add a blue point at mouse when 'b' or 'B' pressed
        if (p.key === 'b' || p.key === 'B') {
          // map mouse coords to feature range [-1, 1]
          const vx = p.map(p.mouseX, 0, p.width, -1, 1);
          const vy = p.map(p.mouseY, p.height, 0, -1, 1);
          const labelSigned = 1; // blue
          const label = 1;
          points.push({ x: vx, y: vy, labelSigned });
          X.push([vx, vy]);
          y.push(label);
          // train on this sample immediately for visible feedback
          try {
            model.trainSample([vx, vy], label);
          } catch (e) {
            // ignore training errors
          }
          return;
        }

        // Add a red point at mouse when lowercase 'r' pressed
        if (p.key === 'r') {
          const vx = p.map(p.mouseX, 0, p.width, -1, 1);
          const vy = p.map(p.mouseY, p.height, 0, -1, 1);
          const labelSigned = -1; // red
          const label = 0; // model uses 0/1
          points.push({ x: vx, y: vy, labelSigned });
          X.push([vx, vy]);
          y.push(label);
          try {
            model.trainSample([vx, vy], label);
          } catch (e) {
            // ignore training errors
          }
          return;
        }

        // Reset dataset/model on uppercase 'R' (preserve quick reset)
        if (p.key === 'R') {
          reset();
          return;
        }

        // speed controls
        if (p.key === '+') {
          speed = Math.min(20, speed + 1);
        } else if (p.key === '-') {
          speed = Math.max(1, speed - 1);
        }
      };
    };

    // If there's an existing p5 instance, remove it first (handles StrictMode double-mount)
    if (p5InstanceRef.current) {
      try {
        p5InstanceRef.current.remove();
      } catch (e) {
        /* ignore */
      }
      p5InstanceRef.current = null;
    }

    // Also ensure the container has no stray canvases
    if (sketchRef.current) {
      const existingCanvases = sketchRef.current.querySelectorAll('canvas');
      existingCanvases.forEach((c) => c.remove());
    }

    const myp5 = new p5(sketch, sketchRef.current);
    p5InstanceRef.current = myp5;
    return () => {
      try {
        if (p5InstanceRef.current) {
          p5InstanceRef.current.remove();
          p5InstanceRef.current = null;
        }
      } catch (e) {
        // ignore removal errors
      }
      // As a fallback, make sure the fixed canvas is removed
      const existing = document.getElementById('perceptron-canvas');
      if (existing) existing.remove();
    };
  }, []);

  return <div ref={sketchRef}></div>;
};

export default PerceptronDemo;
