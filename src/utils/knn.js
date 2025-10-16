// src/utils/KNNClassifier.js

export default class KNNClassifier {
  constructor(k = 5) {
    this.k = k;
    this.points = []; // { x, y, label }
  }

  addSample(x, y, label) {
    this.points.push({ x, y, label });
  }

  distance(a, b) {
    return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2);
  }

  predict(sample) {
    if (this.points.length === 0) return 0;
    const distances = this.points
      .map((p) => ({ dist: this.distance(p, { x: sample[0], y: sample[1] }), label: p.label }))
      .sort((a, b) => a.dist - b.dist)
      .slice(0, this.k);

    const votes = distances.reduce((acc, d) => {
      acc[d.label] = (acc[d.label] || 0) + 1;
      return acc;
    }, {});

    // Majority vote
    let bestLabel = null;
    let bestCount = -1;
    for (const [label, count] of Object.entries(votes)) {
      if (count > bestCount) {
        bestLabel = label;
        bestCount = count;
      }
    }
    return parseInt(bestLabel);
  }
}

/**
 * predictKNN - convenience function used by the UI demo.
 * points: array of { coords: [x,y], label: 'A'|'B'|number }
 * k: number
 * sample: [x,y]
 * returns the predicted label (keeps original label type)
 */
export function predictKNN(points, k, sample) {
  if (!points || points.length === 0) return null;
  // compute distances
  const dists = points.map((p) => ({
    dist: Math.hypot(p.coords[0] - sample[0], p.coords[1] - sample[1]),
    label: p.label,
  }));

  dists.sort((a, b) => a.dist - b.dist);
  const neighbors = dists.slice(0, Math.max(1, Math.min(k, dists.length)));

  const votes = {};
  neighbors.forEach((n) => {
    votes[n.label] = (votes[n.label] || 0) + 1;
  });

  let bestLabel = null;
  let bestCount = -1;
  for (const [label, count] of Object.entries(votes)) {
    if (count > bestCount) {
      bestLabel = label;
      bestCount = count;
    }
  }
  return bestLabel;
}
