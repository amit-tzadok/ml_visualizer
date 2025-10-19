export type UiPoint = { coords: [number, number]; label: string | number };

export function predictKNN(points: UiPoint[] | undefined, k: number, sample: [number, number]) {
  if (!points || points.length === 0) return null;
  const dists = points.map((p) => ({
    dist: Math.hypot(p.coords[0] - sample[0], p.coords[1] - sample[1]),
    label: p.label,
  }));

  dists.sort((a, b) => a.dist - b.dist);
  const neighbors = dists.slice(0, Math.max(1, Math.min(k, dists.length)));

  const votes: Record<string, number> = {};
  neighbors.forEach((n) => {
    const key = String(n.label);
    votes[key] = (votes[key] || 0) + 1;
  });

  let bestLabel: string | null = null;
  let bestCount = -1;
  for (const [label, count] of Object.entries(votes)) {
    if (count > bestCount) {
      bestLabel = label;
      bestCount = count;
    }
  }
  return bestLabel;
}

export default class KNNClassifier {
  k: number;
  points: { x: number; y: number; label: string | number }[];
  constructor(k = 5) {
    this.k = k;
    this.points = [];
  }

  addSample(x: number, y: number, label: string | number) {
    this.points.push({ x, y, label });
  }

  distance(a: { x: number; y: number }, b: { x: number; y: number }) {
    return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2);
  }

  predict(sample: [number, number]) {
    if (this.points.length === 0) return null;
    const distances = this.points
      .map((p) => ({ dist: this.distance(p, { x: sample[0], y: sample[1] }), label: p.label }))
      .sort((a, b) => a.dist - b.dist)
      .slice(0, this.k);

    const votes: Record<string, number> = {};
    distances.forEach((d) => {
      const key = String(d.label);
      votes[key] = (votes[key] || 0) + 1;
    });

    let bestLabel: string | null = null;
    let bestCount = -1;
    for (const [label, count] of Object.entries(votes)) {
      if (count > bestCount) {
        bestLabel = label;
        bestCount = count;
      }
    }
    return bestLabel;
  }
}
