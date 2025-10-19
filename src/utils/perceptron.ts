export default class Perceptron {
  lr: number;
  epochs: number;
  weights: number[];
  bias: number;

  constructor(inputSize: number, learningRate = 0.4, epochs = 100) {
    this.lr = learningRate;
    this.epochs = epochs;
    this.weights = Array.from({ length: inputSize }, () => Math.random());
    this.bias = Math.random();
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

  fitHingeSGD(X: number[][], y: Array<number | string>, options: { epochs?: number; lr?: number; lambda?: number; shuffle?: boolean } = {}) {
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
}
