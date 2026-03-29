/**
 * Decision Tree classifier (CART-style, Gini impurity, binary classification).
 * Supports axis-aligned splits on continuous 2D features.
 */

export interface TreeNode {
  // Split node fields
  feature?: number;
  threshold?: number;
  left?: TreeNode;
  right?: TreeNode;
  // Leaf / metadata
  prediction?: number;
  prob1?: number;    // fraction of class-1 samples at this leaf (for soft heatmap)
  gini?: number;
  samples?: number;
  depth?: number;
}

export interface SplitBoundary {
  feature: number;
  threshold: number;
  depth: number;
  xMin: number;
  xMax: number;
  yMin: number;
  yMax: number;
}

export interface DecisionTreeOptions {
  maxDepth?: number;
  minSamplesSplit?: number;
  minSamplesLeaf?: number;
}

export class DecisionTree {
  maxDepth: number;
  minSamplesSplit: number;
  minSamplesLeaf: number;
  root: TreeNode | null;

  constructor(options: DecisionTreeOptions = {}) {
    this.maxDepth = options.maxDepth ?? 4;
    this.minSamplesSplit = options.minSamplesSplit ?? 4;
    this.minSamplesLeaf = options.minSamplesLeaf ?? 2;
    this.root = null;
  }

  private gini(y: number[]): number {
    if (!y.length) return 0;
    const counts: Record<number, number> = {};
    for (const label of y) counts[label] = (counts[label] ?? 0) + 1;
    let impurity = 1;
    for (const count of Object.values(counts)) {
      const p = count / y.length;
      impurity -= p * p;
    }
    return impurity;
  }

  private majorityClass(y: number[]): number {
    if (!y.length) return 0;
    const counts: Record<number, number> = {};
    for (const label of y) counts[label] = (counts[label] ?? 0) + 1;
    let best = 0;
    let bestCount = -1;
    for (const [label, count] of Object.entries(counts)) {
      if (count > bestCount) { bestCount = count; best = Number(label); }
    }
    return best;
  }

  private bestSplit(
    X: number[][], y: number[]
  ): { feature: number; threshold: number; gain: number } | null {
    const n = X.length;
    if (!n) return null;
    const nFeatures = X[0].length;
    const parentGini = this.gini(y);
    let bestGain = 0;
    let bestFeature = 0;
    let bestThreshold = 0;

    for (let f = 0; f < nFeatures; f++) {
      const rawVals = X.map((row) => row[f]);
      const sorted = [...new Set(rawVals)].sort((a, b) => a - b);

      for (let t = 0; t < sorted.length - 1; t++) {
        const threshold = (sorted[t] + sorted[t + 1]) / 2;
        const leftY: number[] = [];
        const rightY: number[] = [];

        for (let i = 0; i < n; i++) {
          if (X[i][f] <= threshold) leftY.push(y[i]);
          else rightY.push(y[i]);
        }

        if (leftY.length < this.minSamplesLeaf || rightY.length < this.minSamplesLeaf) continue;

        const gain =
          parentGini
          - (leftY.length / n) * this.gini(leftY)
          - (rightY.length / n) * this.gini(rightY);

        if (gain > bestGain) {
          bestGain = gain;
          bestFeature = f;
          bestThreshold = threshold;
        }
      }
    }

    return bestGain > 0 ? { feature: bestFeature, threshold: bestThreshold, gain: bestGain } : null;
  }

  private buildNode(X: number[][], y: number[], depth: number): TreeNode {
    const node: TreeNode = { gini: this.gini(y), samples: y.length, depth };
    // Store class-1 fraction for soft probability heatmap
    node.prob1 = y.length ? y.filter(v => v === 1).length / y.length : 0.5;

    // Leaf conditions
    if (
      depth >= this.maxDepth ||
      y.length < this.minSamplesSplit ||
      this.gini(y) === 0
    ) {
      node.prediction = this.majorityClass(y);
      return node;
    }

    const split = this.bestSplit(X, y);
    if (!split) {
      node.prediction = this.majorityClass(y);
      return node;
    }

    const { feature, threshold } = split;
    node.feature = feature;
    node.threshold = threshold;

    const leftX: number[][] = [];
    const leftY: number[] = [];
    const rightX: number[][] = [];
    const rightY: number[] = [];

    for (let i = 0; i < X.length; i++) {
      if (X[i][feature] <= threshold) {
        leftX.push(X[i]); leftY.push(y[i]);
      } else {
        rightX.push(X[i]); rightY.push(y[i]);
      }
    }

    node.left = this.buildNode(leftX, leftY, depth + 1);
    node.right = this.buildNode(rightX, rightY, depth + 1);
    return node;
  }

  fit(X: number[][], y: number[]): void {
    if (!X.length) return;
    this.root = this.buildNode(X, y, 0);
  }

  predictSample(x: number[]): number {
    let node: TreeNode | null | undefined = this.root;
    while (node) {
      if (node.prediction !== undefined) return node.prediction;
      if (node.feature === undefined || node.threshold === undefined) return 0;
      node = x[node.feature] <= node.threshold ? node.left : node.right;
    }
    return 0;
  }

  /** Soft probability of class 1 (fraction of class-1 samples at the leaf). */
  predictProbaSample(x: number[]): number {
    let node: TreeNode | null | undefined = this.root;
    while (node) {
      if (node.prediction !== undefined) return node.prob1 ?? (node.prediction === 1 ? 1 : 0);
      if (node.feature === undefined || node.threshold === undefined) return 0.5;
      node = x[node.feature] <= node.threshold ? node.left : node.right;
    }
    return 0.5;
  }

  predict(X: number[][]): number[] {
    return X.map((x) => this.predictSample(x));
  }

  accuracy(X: number[][], y: number[]): number {
    if (!X.length) return 0;
    return this.predict(X).filter((p, i) => p === y[i]).length / y.length;
  }

  /**
   * Collect all split boundaries as axis-aligned segments for 2D visualization.
   * Each boundary has the spatial extent of its parent region.
   */
  getSplitBoundaries(
    xRange: [number, number] = [-1, 1],
    yRange: [number, number] = [-1, 1],
    depth = 0,
    node: TreeNode | null | undefined = this.root
  ): SplitBoundary[] {
    if (!node || node.prediction !== undefined) return [];
    if (node.feature === undefined || node.threshold === undefined) return [];

    const boundaries: SplitBoundary[] = [
      {
        feature: node.feature,
        threshold: node.threshold,
        depth,
        xMin: xRange[0], xMax: xRange[1],
        yMin: yRange[0], yMax: yRange[1],
      },
    ];

    if (node.feature === 0) {
      // Vertical split on x
      boundaries.push(
        ...this.getSplitBoundaries([xRange[0], node.threshold], yRange, depth + 1, node.left),
        ...this.getSplitBoundaries([node.threshold, xRange[1]], yRange, depth + 1, node.right),
      );
    } else {
      // Horizontal split on y
      boundaries.push(
        ...this.getSplitBoundaries(xRange, [yRange[0], node.threshold], depth + 1, node.left),
        ...this.getSplitBoundaries(xRange, [node.threshold, yRange[1]], depth + 1, node.right),
      );
    }

    return boundaries;
  }

  /** Tree depth (0 = no tree, 1 = root only / single leaf) */
  treeDepth(node: TreeNode | null | undefined = this.root): number {
    if (!node) return 0;
    if (node.prediction !== undefined) return 1;
    return 1 + Math.max(
      this.treeDepth(node.left),
      this.treeDepth(node.right),
    );
  }

  reset(): void {
    this.root = null;
  }
}

export default DecisionTree;
