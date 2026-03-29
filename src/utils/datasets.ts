/**
 * Standard ML benchmark datasets for 2D binary classification.
 * All coordinates are in the range [-1.2, 1.2] approximately.
 */

export interface DataPoint {
  x: number;
  y: number;
  label: 0 | 1;
}

/** Box-Muller Gaussian noise */
function gaussianNoise(std: number): number {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v) * std;
}

/**
 * XOR dataset — four quadrants with alternating labels.
 * Requires a nonlinear classifier (e.g. MLP, polynomial, decision tree).
 */
export function makeXOR(n = 200, noise = 0.08): DataPoint[] {
  const pts: DataPoint[] = [];
  for (let i = 0; i < n; i++) {
    const x = (Math.random() * 2 - 1) * 0.9;
    const y = (Math.random() * 2 - 1) * 0.9;
    const label: 0 | 1 = (x > 0) === (y > 0) ? 1 : 0;
    pts.push({ x: x + gaussianNoise(noise), y: y + gaussianNoise(noise), label });
  }
  return pts;
}

/**
 * Two-moons dataset — two interleaved crescent shapes.
 * Classic benchmark; linearly inseparable.
 */
export function makeMoons(n = 200, noise = 0.1): DataPoint[] {
  const pts: DataPoint[] = [];
  const half = Math.floor(n / 2);

  for (let i = 0; i < half; i++) {
    const angle = (Math.PI * i) / half;
    pts.push({
      x: Math.cos(angle) * 0.8 + gaussianNoise(noise),
      y: Math.sin(angle) * 0.8 + gaussianNoise(noise),
      label: 0,
    });
  }

  for (let i = 0; i < n - half; i++) {
    const angle = (Math.PI * i) / (n - half);
    pts.push({
      x: (1 - Math.cos(angle)) * 0.8 - 0.8 + gaussianNoise(noise),
      y: -Math.sin(angle) * 0.8 + 0.3 + gaussianNoise(noise),
      label: 1,
    });
  }

  return pts;
}

/**
 * Concentric circles dataset — inner vs. outer ring.
 * factor controls inner ring radius relative to outer.
 */
export function makeCircles(n = 200, noise = 0.05, factor = 0.4): DataPoint[] {
  const pts: DataPoint[] = [];
  const half = Math.floor(n / 2);

  for (let i = 0; i < half; i++) {
    const angle = (2 * Math.PI * i) / half;
    pts.push({
      x: Math.cos(angle) * 0.9 + gaussianNoise(noise),
      y: Math.sin(angle) * 0.9 + gaussianNoise(noise),
      label: 0,
    });
  }

  for (let i = 0; i < n - half; i++) {
    const angle = (2 * Math.PI * i) / (n - half);
    pts.push({
      x: Math.cos(angle) * 0.9 * factor + gaussianNoise(noise),
      y: Math.sin(angle) * 0.9 * factor + gaussianNoise(noise),
      label: 1,
    });
  }

  return pts;
}

/**
 * Gaussian blobs — linearly separable, two clusters.
 */
export function makeBlobs(n = 200, noise = 0.12): DataPoint[] {
  const pts: DataPoint[] = [];
  const half = Math.floor(n / 2);

  for (let i = 0; i < half; i++) {
    pts.push({ x: -0.5 + gaussianNoise(noise), y: -0.3 + gaussianNoise(noise), label: 0 });
  }

  for (let i = 0; i < n - half; i++) {
    pts.push({ x: 0.5 + gaussianNoise(noise), y: 0.3 + gaussianNoise(noise), label: 1 });
  }

  return pts;
}

/**
 * Two-spiral dataset — two interleaved spirals, very nonlinear.
 */
export function makeSpirals(n = 200, noise = 0.04): DataPoint[] {
  const pts: DataPoint[] = [];
  const half = Math.floor(n / 2);

  for (let i = 0; i < half; i++) {
    const t = (i / half) * 3 * Math.PI;
    const r = (i / half) * 0.85;
    pts.push({
      x: r * Math.cos(t) + gaussianNoise(noise),
      y: r * Math.sin(t) + gaussianNoise(noise),
      label: 0,
    });
  }

  for (let i = 0; i < n - half; i++) {
    const t = (i / (n - half)) * 3 * Math.PI + Math.PI;
    const r = (i / (n - half)) * 0.85;
    pts.push({
      x: r * Math.cos(t) + gaussianNoise(noise),
      y: r * Math.sin(t) + gaussianNoise(noise),
      label: 1,
    });
  }

  return pts;
}

/**
 * Convert a DataPoint array to the [X, y] matrix format used by algorithm utilities.
 */
export function toXy(pts: DataPoint[]): { X: number[][]; y: number[] } {
  return {
    X: pts.map((p) => [p.x, p.y]),
    y: pts.map((p) => p.label),
  };
}
