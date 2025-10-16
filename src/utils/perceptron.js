
export default class Perceptron {
  constructor(inputSize, learningRate = 0.4, epochs = 100) {
    this.lr = learningRate;
    this.epochs = epochs;
    this.weights = Array.from({ length: inputSize }, () => Math.random());
    this.bias = Math.random();
  }

  activation(x) {
    // Step function, threshold at 0
    return x >= 0 ? 1 : 0;
  }

  predict(x) {
    // Dot product of weights and input + bias
    let linearOutput = this.bias;
    for (let i = 0; i < this.weights.length; i++) {
      linearOutput += this.weights[i] * x[i];
    }
    return this.activation(linearOutput);
  }

  // Return raw linear output (before activation). Useful for losses that
  // depend on the real-valued score (hinge loss, MSE, etc.).
  predictRaw(x) {
    let linearOutput = this.bias;
    for (let i = 0; i < this.weights.length; i++) {
      linearOutput += this.weights[i] * x[i];
    }
    return linearOutput;
  }

  fit(X, y) {
    for (let epoch = 0; epoch < this.epochs; epoch++) {
      for (let i = 0; i < X.length; i++) {
        this.trainSample(X[i], y[i]);
      }
    }
  }

  // Train on a single sample (useful for incremental / live visualizations)
  trainSample(x, y) {
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

  // Compute simple evaluation metrics on arrays of samples.
  // X: array of input vectors, y: array of labels (either 0/1 or -1/1)
  misclassificationRate(X, y) {
    if (!Array.isArray(X) || !Array.isArray(y) || X.length !== y.length)
      return NaN;
    let mismatches = 0;
    for (let i = 0; i < X.length; i++) {
      const pred = this.predict(X[i]);
      const yi = y[i] === 1 || y[i] === "1" ? 1 : 0;
      if (pred !== yi) mismatches++;
    }
    return mismatches / X.length;
  }

  // Hinge loss averaged over samples. Expects labels in 0/1 or -1/1; converts to -1/1.
  hingeLoss(X, y) {
    if (!Array.isArray(X) || !Array.isArray(y) || X.length !== y.length)
      return NaN;
    let total = 0;
    for (let i = 0; i < X.length; i++) {
      const yi = y[i] === 1 || y[i] === "1" ? 1 : -1;
      const score = this.predictRaw(X[i]);
      total += Math.max(0, 1 - yi * score);
    }
    return total / X.length;
  }

  // Mean squared error on the raw score vs. target (target treated as 0/1)
  mseRaw(X, y) {
    if (!Array.isArray(X) || !Array.isArray(y) || X.length !== y.length)
      return NaN;
    let total = 0;
    for (let i = 0; i < X.length; i++) {
      const target = y[i] === 1 || y[i] === "1" ? 1 : 0;
      const score = this.predictRaw(X[i]);
      const err = score - target;
      total += err * err;
    }
    return total / X.length;
  }

  // Train using hinge-loss SGD (primal SVM style) with L2 regularization.
  // Options: { epochs=10, lr=0.01, lambda=0.01, shuffle=true }
  // Labels y should be 0/1 or -1/1; converted to -1/+1 internally.
  fitHingeSGD(X, y, options = {}) {
    const { epochs = 10, lr = 0.01, lambda = 0.01, shuffle = true } = options;
    if (!Array.isArray(X) || !Array.isArray(y) || X.length !== y.length) return;

    // convert labels to -1/+1
    const Y = y.map((v) => (v === 1 || v === "1" ? 1 : -1));

    const n = X.length;
    // safety: ensure weights length matches input length
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
        // score = w·x + b
        const score = this.predictRaw(xi);
        if (yi * score < 1) {
          // gradient: -yi * xi + lambda * w
          for (let j = 0; j < this.weights.length; j++) {
            this.weights[j] = this.weights[j] * (1 - lr * lambda) + lr * yi * xi[j];
          }
          // bias update (no regularization on bias)
          this.bias = this.bias + lr * yi;
        } else {
          // only regularization term
          for (let j = 0; j < this.weights.length; j++) {
            this.weights[j] = this.weights[j] * (1 - lr * lambda);
          }
        }
      }
    }
  }
}
