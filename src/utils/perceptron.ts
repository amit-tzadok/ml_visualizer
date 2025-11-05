export type PerceptronLabel = 0 | 1 | -1 | "0" | "1" | "A" | "B";

export interface FitOptions {
  epochs?: number;
  shuffle?: boolean;
  earlyStop?: {
    patience?: number; // number of epochs without improvement before stop
    metric?: "errors" | "hinge"; // which metric to monitor
  };
  lrSchedule?: (epoch: number, baseLr: number) => number; // optional LR schedule
}

export interface PocketOptions extends FitOptions {
  maxUpdates?: number; // hard cap on updates for non-separable sets
}

export default class Perceptron {
  lr: number;
  epochs: number;
  weights: number[];
  bias: number;

  constructor(inputSize: number, learningRate = 0.4, epochs = 100) {
    this.lr = learningRate;
    this.epochs = epochs;
    // Initialize near zero for stability across demos
    this.weights = Array.from({ length: inputSize }, () => (Math.random() - 0.5) * 0.02);
    this.bias = (Math.random() - 0.5) * 0.02;
  }

  activation(x: number) {
    return x >= 0 ? 1 : 0;
  }

  predict(x: number[]) {
    let linearOutput = this.bias;
    for (let i = 0; i < this.weights.length; i++) {
      linearOutput += this.weights[i] * x[i];
    }
    return this.activation(linearOutput);
  }

  predictRaw(x: number[]) {
    let linearOutput = this.bias;
    for (let i = 0; i < this.weights.length; i++) {
      linearOutput += this.weights[i] * x[i];
    }
    return linearOutput;
  }

  fit(X: number[][], y: Array<number | string>) {
    for (let epoch = 0; epoch < this.epochs; epoch++) {
      for (let i = 0; i < X.length; i++) {
        this.trainSample(X[i], Number(y[i]));
      }
    }
  }

  trainSample(x: number[], y: number) {
    if (!Array.isArray(x) || x.length !== this.weights.length) {
      throw new Error(`Input vector length (${x && x.length}) does not match weights length (${this.weights.length}).`);
    }
    const prediction = this.predict(x);
    const error = y - prediction;
    for (let j = 0; j < this.weights.length; j++) {
      this.weights[j] += this.lr * error * x[j];
    }
    this.bias += this.lr * error;
  }

  misclassificationRate(X: number[][], y: Array<number | string>) {
    if (!Array.isArray(X) || !Array.isArray(y) || X.length !== y.length) return NaN;
    let mismatches = 0;
    for (let i = 0; i < X.length; i++) {
      const pred = this.predict(X[i]);
      const yi = y[i] === 1 || y[i] === "1" ? 1 : 0;
      if (pred !== yi) mismatches++;
    }
    return mismatches / X.length;
  }

  hingeLoss(X: number[][], y: Array<number | string>) {
    if (!Array.isArray(X) || !Array.isArray(y) || X.length !== y.length) return NaN;
    let total = 0;
    for (let i = 0; i < X.length; i++) {
      const yi = y[i] === 1 || y[i] === "1" ? 1 : -1;
      const score = this.predictRaw(X[i]);
      total += Math.max(0, 1 - yi * score);
    }
    return total / X.length;
  }

  mseRaw(X: number[][], y: Array<number | string>) {
    if (!Array.isArray(X) || !Array.isArray(y) || X.length !== y.length) return NaN;
    let total = 0;
    for (let i = 0; i < X.length; i++) {
      const target = y[i] === 1 || y[i] === "1" ? 1 : 0;
      const score = this.predictRaw(X[i]);
      const err = score - target;
      total += err * err;
    }
    return total / X.length;
  }

  fitHingeSGD(
    X: number[][],
    y: Array<number | string>,
    options: { epochs?: number; lr?: number; lambda?: number; shuffle?: boolean } = {}
  ) {
    const { epochs = 10, lr = 0.01, lambda = 0.01, shuffle = true } = options;
    if (!Array.isArray(X) || !Array.isArray(y) || X.length !== y.length) return;

    const Y = y.map((v) => (v === 1 || v === "1" ? 1 : -1));

    const n = X.length;
    if (!this.weights || this.weights.length === 0) {
      this.weights = Array.from({ length: X[0].length }, () => Math.random() - 0.5);
    }

    const idx = Array.from({ length: n }, (_, i) => i);
    for (let epoch = 0; epoch < epochs; epoch++) {
      if (shuffle) {
        for (let i = n - 1; i > 0; i--) {
          const j = Math.floor(Math.random() * (i + 1));
          const tmp = idx[i];
          idx[i] = idx[j];
          idx[j] = tmp;
        }
      }

      for (let t = 0; t < n; t++) {
        const i = idx[t];
        const xi = X[i];
        const yi = Y[i];
        const score = this.predictRaw(xi);
        if (yi * score < 1) {
          for (let j = 0; j < this.weights.length; j++) {
            this.weights[j] = this.weights[j] * (1 - lr * lambda) + lr * yi * xi[j];
          }
          this.bias = this.bias + lr * yi;
        } else {
          for (let j = 0; j < this.weights.length; j++) {
            this.weights[j] = this.weights[j] * (1 - lr * lambda);
          }
        }
      }
    }
  }

  /** Map a label to {0,1} or {-1,1} depending on requested space */
  private mapLabel(y: PerceptronLabel, space: "01" | "+-" = "01"): 0 | 1 | -1 {
    if (space === "01") {
      if (y === 1 || y === "1" || y === "A") return 1;
      return 0;
    }
    // space '+-' (aka {-1, +1})
    if (y === 1 || y === "1" || y === "A") return 1;
    return -1;
  }

  /** Online perceptron with shuffling, early stopping and optional LR schedule */
  fitOnline(
    X: number[][],
    y: Array<number | string>,
    opts: FitOptions = {}
  ) {
    const epochs = opts.epochs ?? this.epochs;
    const shuffle = opts.shuffle ?? true;
    const patience = opts.earlyStop?.patience ?? 0;
    const metric = opts.earlyStop?.metric ?? "errors";
    const lrSchedule = opts.lrSchedule;

    if (!Array.isArray(X) || !Array.isArray(y) || X.length !== y.length || X.length === 0) return;

    let bestMetric = Infinity;
    let bestW = this.weights.slice();
    let bestB = this.bias;
    let epochsSinceBest = 0;

    const idx = Array.from({ length: X.length }, (_, i) => i);
    for (let epoch = 0; epoch < epochs; epoch++) {
      const lr = Math.max(1e-5, lrSchedule ? lrSchedule(epoch, this.lr) : this.lr);
      if (shuffle) {
        for (let i = idx.length - 1; i > 0; i--) {
          const j = Math.floor(Math.random() * (i + 1));
          const t = idx[i];
          idx[i] = idx[j];
          idx[j] = t;
        }
      }
      let mistakes = 0;
      for (let t = 0; t < idx.length; t++) {
        const i = idx[t];
        const xi = X[i];
  const yi = this.mapLabel(y[i] as PerceptronLabel, "01");
        const pred = this.predict(xi);
        const err = yi - pred;
        if (err !== 0) mistakes++;
        for (let j = 0; j < this.weights.length; j++) {
          this.weights[j] += lr * err * xi[j];
        }
        this.bias += lr * err;
      }
      const currentMetric = metric === "hinge" ? this.hingeLoss(X, y) : mistakes;
      if (currentMetric < bestMetric - 1e-12) {
        bestMetric = currentMetric;
        bestW = this.weights.slice();
        bestB = this.bias;
        epochsSinceBest = 0;
      } else {
        epochsSinceBest++;
        if (patience > 0 && epochsSinceBest >= patience) break;
      }
      if (mistakes === 0) break; // converged on separable data
    }
    this.weights = bestW;
    this.bias = bestB;
  }

  /** Averaged perceptron — returns averaged params for better generalization */
  fitAveraged(
    X: number[][],
    y: Array<number | string>,
    opts: FitOptions = {}
  ) {
    const epochs = opts.epochs ?? this.epochs;
    const shuffle = opts.shuffle ?? true;
    if (!Array.isArray(X) || !Array.isArray(y) || X.length !== y.length || X.length === 0) return;

    const idx = Array.from({ length: X.length }, (_, i) => i);
    const w = this.weights.slice();
    let b = this.bias;
    const cW = new Array(w.length).fill(0);
    let cB = 0;
    let count = 0;

    for (let epoch = 0; epoch < epochs; epoch++) {
      if (shuffle) {
        for (let i = idx.length - 1; i > 0; i--) {
          const j = Math.floor(Math.random() * (i + 1));
          const t = idx[i];
          idx[i] = idx[j];
          idx[j] = t;
        }
      }
      for (let t = 0; t < idx.length; t++) {
        const i = idx[t];
        const xi = X[i];
  const yi = this.mapLabel(y[i] as PerceptronLabel, "+-"); // averaged perceptron uses {-1,+1}
        const score = b + w.reduce((s, wj, j) => s + wj * xi[j], 0);
        if (yi * score <= 0) {
          for (let j = 0; j < w.length; j++) w[j] += this.lr * yi * xi[j];
          b += this.lr * yi;
        }
        // accumulate
        for (let j = 0; j < w.length; j++) cW[j] += w[j];
        cB += b;
        count++;
      }
    }
    // set to averaged params
    this.weights = cW.map((v) => v / Math.max(1, count));
    this.bias = cB / Math.max(1, count);
  }

  /** Pocket algorithm — keep best weights under misclassification metric (useful with noise) */
  fitPocket(
    X: number[][],
    y: Array<number | string>,
    opts: PocketOptions = {}
  ) {
    const epochs = opts.epochs ?? this.epochs;
    const shuffle = opts.shuffle ?? true;
    const maxUpdates = opts.maxUpdates ?? epochs * X.length;
    if (!Array.isArray(X) || !Array.isArray(y) || X.length !== y.length || X.length === 0) return;

    const idx = Array.from({ length: X.length }, (_, i) => i);
    let bestW = this.weights.slice();
    let bestB = this.bias;
    let bestErr = this.misclassificationRate(X, y);
    let updates = 0;

    for (let epoch = 0; epoch < epochs && updates < maxUpdates; epoch++) {
      if (shuffle) {
        for (let i = idx.length - 1; i > 0; i--) {
          const j = Math.floor(Math.random() * (i + 1));
          const t = idx[i];
          idx[i] = idx[j];
          idx[j] = t;
        }
      }
      for (let t = 0; t < idx.length && updates < maxUpdates; t++) {
        const i = idx[t];
        const xi = X[i];
  const yi01 = this.mapLabel(y[i] as PerceptronLabel, "01");
        const pred = this.predict(xi);
        const err = yi01 - pred;
        if (err !== 0) {
          for (let j = 0; j < this.weights.length; j++) this.weights[j] += this.lr * err * xi[j];
          this.bias += this.lr * err;
          updates++;
          const e = this.misclassificationRate(X, y);
          if (e < bestErr) {
            bestErr = e;
            bestW = this.weights.slice();
            bestB = this.bias;
          }
        }
      }
    }
    this.weights = bestW;
    this.bias = bestB;
  }
}
