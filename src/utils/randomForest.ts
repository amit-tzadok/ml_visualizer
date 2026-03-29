/**
 * Random Forest classifier — ensemble of CART decision trees.
 * Each tree is trained on a bootstrap sample of the training data.
 * Prediction is by majority vote (or probability averaging).
 */

import DecisionTree from './decisionTree';

export interface RandomForestOptions {
  nEstimators?: number;   // number of trees (default 20)
  maxDepth?: number;      // max depth per tree (default 5)
  minSamplesSplit?: number;
  minSamplesLeaf?: number;
}

export class RandomForest {
  nEstimators: number;
  maxDepth: number;
  minSamplesSplit: number;
  minSamplesLeaf: number;
  trees: DecisionTree[];

  constructor(options: RandomForestOptions = {}) {
    this.nEstimators = options.nEstimators ?? 20;
    this.maxDepth = options.maxDepth ?? 5;
    this.minSamplesSplit = options.minSamplesSplit ?? 4;
    this.minSamplesLeaf = options.minSamplesLeaf ?? 2;
    this.trees = [];
  }

  /** Train all trees on bootstrap samples. */
  fit(X: number[][], y: number[]): void {
    this.trees = [];
    const n = X.length;
    if (!n) return;

    for (let t = 0; t < this.nEstimators; t++) {
      const tree = new DecisionTree({
        maxDepth: this.maxDepth,
        minSamplesSplit: this.minSamplesSplit,
        minSamplesLeaf: this.minSamplesLeaf,
      });

      // Bootstrap sample (sample with replacement)
      const bootX: number[][] = [];
      const bootY: number[] = [];
      for (let i = 0; i < n; i++) {
        const idx = Math.floor(Math.random() * n);
        bootX.push(X[idx]);
        bootY.push(y[idx]);
      }
      tree.fit(bootX, bootY);
      this.trees.push(tree);
    }
  }

  /** Add a single tree trained on a bootstrap sample (for incremental animation). */
  addTree(X: number[][], y: number[]): void {
    const n = X.length;
    if (!n) return;
    const tree = new DecisionTree({
      maxDepth: this.maxDepth,
      minSamplesSplit: this.minSamplesSplit,
      minSamplesLeaf: this.minSamplesLeaf,
    });
    const bootX: number[][] = [];
    const bootY: number[] = [];
    for (let i = 0; i < n; i++) {
      const idx = Math.floor(Math.random() * n);
      bootX.push(X[idx]);
      bootY.push(y[idx]);
    }
    tree.fit(bootX, bootY);
    this.trees.push(tree);
  }

  /** Predict class (majority vote). */
  predictSample(x: number[]): number {
    if (!this.trees.length) return 0;
    const votes = this.trees.reduce((sum, t) => sum + t.predictSample(x), 0);
    return votes / this.trees.length >= 0.5 ? 1 : 0;
  }

  /** Predict probability of class 1 (vote fraction). */
  predictProba(x: number[]): number {
    if (!this.trees.length) return 0.5;
    return this.trees.reduce((sum, t) => sum + t.predictSample(x), 0) / this.trees.length;
  }

  predict(X: number[][]): number[] {
    return X.map(x => this.predictSample(x));
  }

  accuracy(X: number[][], y: number[]): number {
    if (!X.length || !this.trees.length) return 0;
    return this.predict(X).filter((p, i) => p === y[i]).length / y.length;
  }

  reset(): void {
    this.trees = [];
  }
}

export default RandomForest;
