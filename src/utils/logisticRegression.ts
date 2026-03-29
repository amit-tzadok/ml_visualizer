/**
 * Logistic Regression with mini-batch gradient descent (binary classification).
 * Features L2 regularization, learning rate scheduling, and early stopping.
 */

export interface LogisticRegressionOptions {
  lr?: number;
  lambda?: number;     // L2 regularization coefficient
  batchSize?: number;
  lrDecay?: number;    // multiplicative LR decay per epoch (e.g. 0.99)
  tolerance?: number;  // early stopping on loss improvement
}

export interface TrainRecord {
  epoch: number;
  loss: number;
  accuracy: number;
}

export class LogisticRegression {
  weights: number[];
  bias: number;
  lr: number;
  lambda: number;
  batchSize: number;
  lrDecay: number;
  tolerance: number;
  trainHistory: TrainRecord[];

  constructor(inputSize: number, options: LogisticRegressionOptions = {}) {
    this.lr = options.lr ?? 0.1;
    this.lambda = options.lambda ?? 0.001;
    this.batchSize = options.batchSize ?? 16;
    this.lrDecay = options.lrDecay ?? 1.0;
    this.tolerance = options.tolerance ?? 1e-7;
    this.trainHistory = [];
    // Xavier initialization: std = 1/sqrt(inputSize)
    const std = Math.sqrt(1 / Math.max(1, inputSize));
    this.weights = Array.from({ length: inputSize }, () => (Math.random() * 2 - 1) * std);
    this.bias = 0;
  }

  static sigmoid(z: number): number {
    // Numerically stable sigmoid avoids overflow for large |z|
    if (z >= 0) return 1 / (1 + Math.exp(-z));
    const ez = Math.exp(z);
    return ez / (1 + ez);
  }

  private score(x: number[]): number {
    let z = this.bias;
    for (let i = 0; i < this.weights.length; i++) z += this.weights[i] * x[i];
    return z;
  }

  /** Sigmoid probability P(y=1 | x) */
  probability(x: number[]): number {
    return LogisticRegression.sigmoid(this.score(x));
  }

  /** Binary prediction (0 or 1) */
  predict(x: number[]): number {
    return this.probability(x) >= 0.5 ? 1 : 0;
  }

  /** Binary cross-entropy loss + L2 regularization */
  loss(X: number[][], y: number[]): number {
    if (!X.length) return 0;
    let total = 0;
    for (let i = 0; i < X.length; i++) {
      const p = Math.min(1 - 1e-7, Math.max(1e-7, this.probability(X[i])));
      total -= y[i] * Math.log(p) + (1 - y[i]) * Math.log(1 - p);
    }
    // L2 penalty: lambda/2 * ||w||^2
    const reg = (this.lambda / 2) * this.weights.reduce((s, w) => s + w * w, 0);
    return total / X.length + reg;
  }

  accuracy(X: number[][], y: number[]): number {
    if (!X.length) return 0;
    let correct = 0;
    for (let i = 0; i < X.length; i++) {
      if (this.predict(X[i]) === y[i]) correct++;
    }
    return correct / X.length;
  }

  /**
   * Train with mini-batch gradient descent.
   * Returns loss history (one value per epoch).
   */
  fit(
    X: number[][],
    y: number[],
    options: LogisticRegressionOptions & {
      epochs?: number;
      onEpoch?: (epoch: number, loss: number) => void;
    } = {}
  ): number[] {
    const epochs = options.epochs ?? 100;
    const lr = options.lr ?? this.lr;
    const batchSize = options.batchSize ?? this.batchSize;
    const onEpoch = options.onEpoch;

    this.trainHistory = [];
    const n = X.length;
    const lossHistory: number[] = [];
    const idx = Array.from({ length: n }, (_, i) => i);
    let currentLr = lr;
    let prevLoss = Infinity;

    for (let epoch = 0; epoch < epochs; epoch++) {
      // Fisher-Yates shuffle
      for (let i = n - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        const tmp = idx[i]; idx[i] = idx[j]; idx[j] = tmp;
      }

      for (let b = 0; b < n; b += batchSize) {
        const end = Math.min(b + batchSize, n);
        const batchN = end - b;
        const gradW = new Array(this.weights.length).fill(0);
        let gradB = 0;

        for (let t = b; t < end; t++) {
          const xi = X[idx[t]];
          const yi = y[idx[t]];
          const err = this.probability(xi) - yi;  // d(BCE)/dz = p - y
          for (let j = 0; j < this.weights.length; j++) gradW[j] += err * xi[j];
          gradB += err;
        }

        for (let j = 0; j < this.weights.length; j++) {
          this.weights[j] -= currentLr * (gradW[j] / batchN + this.lambda * this.weights[j]);
        }
        this.bias -= currentLr * (gradB / batchN);
      }

      const epochLoss = this.loss(X, y);
      const epochAcc = this.accuracy(X, y);
      lossHistory.push(epochLoss);
      this.trainHistory.push({ epoch, loss: epochLoss, accuracy: epochAcc });

      if (typeof onEpoch === 'function') {
        try { onEpoch(epoch, epochLoss); } catch { /* ignore */ }
      }

      if (Math.abs(prevLoss - epochLoss) < this.tolerance) break;
      prevLoss = epochLoss;
      currentLr *= this.lrDecay;
    }

    return lossHistory;
  }

  /**
   * Partial fit — one shuffled pass (for interactive animation).
   * Returns the loss after the update.
   */
  fitStep(X: number[][], y: number[], lr?: number): number {
    const eta = lr ?? this.lr;
    const n = X.length;
    if (!n) return Infinity;

    const idx = Array.from({ length: n }, (_, i) => i);
    for (let i = n - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      const tmp = idx[i]; idx[i] = idx[j]; idx[j] = tmp;
    }

    for (let t = 0; t < n; t++) {
      const xi = X[idx[t]];
      const yi = y[idx[t]];
      const err = this.probability(xi) - yi;
      for (let j = 0; j < this.weights.length; j++) {
        this.weights[j] -= eta * (err * xi[j] + this.lambda * this.weights[j]);
      }
      this.bias -= eta * err;
    }

    return this.loss(X, y);
  }

  reset(inputSize?: number): void {
    const size = inputSize ?? this.weights.length;
    const std = Math.sqrt(1 / Math.max(1, size));
    this.weights = Array.from({ length: size }, () => (Math.random() * 2 - 1) * std);
    this.bias = 0;
    this.trainHistory = [];
  }

  /**
   * Decision boundary as slope and intercept for 2D input.
   * Returns null if input is not 2D or weight[1] is near zero.
   * In feature space: w0*x + w1*y + b = 0  =>  y = -(w0/w1)*x - b/w1
   */
  decisionBoundary2D(): { slope: number; intercept: number } | null {
    if (this.weights.length !== 2) return null;
    const [w0, w1] = this.weights;
    if (Math.abs(w1) < 1e-10) return null;
    return { slope: -w0 / w1, intercept: -this.bias / w1 };
  }
}

export default LogisticRegression;
