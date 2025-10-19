// Lightweight vanilla MLP classifier (binary) - JS test implementation
// This file mirrors the TS implementation but is plain JS to avoid
// test-runtime transform issues in some environments.

export default class MLPClassifier {
  constructor(inputSize = 2, hidden = [16], options = {}) {
    this.inputSize = inputSize;
    this.hidden = Array.isArray(hidden) ? hidden : [hidden];
    this.lr = options.lr || 0.01;
    this.initWeights();
  }

  initWeights() {
    this.weights = [];
    this.biases = [];
    let prev = this.inputSize;
    for (let i = 0; i < this.hidden.length; i++) {
      const h = this.hidden[i];
      this.weights.push(Array.from({ length: h }, () =>
        Array.from({ length: prev }, () => (Math.random() - 0.5) * 0.5)
      ));
      this.biases.push(Array.from({ length: h }, () => 0));
      prev = h;
    }
    this.weights.push(Array.from({ length: 1 }, () =>
      Array.from({ length: prev }, () => (Math.random() - 0.5) * 0.5)
    ));
    this.biases.push([0]);
  }

  static tanh(x) { return Math.tanh(x); }
  static tanhDeriv(y) { return 1 - y * y; }
  static sigmoid(x) { return 1 / (1 + Math.exp(-x)); }

  forward(x) {
    let a = x.slice();
    for (let l = 0; l < this.weights.length; l++) {
      const W = this.weights[l];
      const b = this.biases[l];
      const z = new Array(W.length).fill(0);
      for (let j = 0; j < W.length; j++) {
        let s = b[j] || 0;
        const row = W[j];
        for (let k = 0; k < row.length; k++) s += row[k] * a[k];
        z[j] = s;
      }
      if (l < this.weights.length - 1) a = z.map((v) => MLPClassifier.tanh(v));
      else a = z.map((v) => MLPClassifier.sigmoid(v));
    }
    return a[0];
  }

  predict(x) { return this.forward(x) >= 0.5 ? 1 : 0; }

  trainSample(x, y, lr = null) {
    const eta = lr == null ? this.lr : lr;
    const activations = [x.slice()];
    let a = x.slice();
    const zs = [];
    for (let l = 0; l < this.weights.length; l++) {
      const W = this.weights[l];
      const b = this.biases[l];
      const z = new Array(W.length).fill(0);
      for (let j = 0; j < W.length; j++) {
        let s = b[j] || 0;
        const row = W[j];
        for (let k = 0; k < row.length; k++) s += row[k] * a[k];
        z[j] = s;
      }
      zs.push(z);
      if (l < this.weights.length - 1) a = z.map((v) => MLPClassifier.tanh(v));
      else a = z.map((v) => MLPClassifier.sigmoid(v));
      activations.push(a);
    }

    const pred = activations[activations.length - 1][0];
    const delta_out = pred - y;
    let delta = [delta_out];
    for (let l = this.weights.length - 1; l >= 0; l--) {
      const a_prev = activations[l];
      for (let j = 0; j < this.weights[l].length; j++) {
        const grad = delta[j];
        for (let k = 0; k < this.weights[l][j].length; k++) {
          this.weights[l][j][k] -= eta * grad * a_prev[k];
        }
        this.biases[l][j] -= eta * grad;
      }
      if (l > 0) {
        const nextDelta = new Array(this.weights[l - 1].length).fill(0);
        for (let i = 0; i < this.weights[l].length; i++) {
          const row = this.weights[l][i];
          for (let k = 0; k < row.length; k++) {
            nextDelta[k] += row[k] * delta[i];
          }
        }
        delta = nextDelta.map((v, idx) => v * MLPClassifier.tanhDeriv(activations[l][idx]));
      }
    }
  }

  fit(X, y, options = {}) {
    const epochs = options.epochs || 20;
    const lr = options.lr || this.lr;
    const batchSize = options.batchSize || 8;
    const shuffle = options.shuffle !== undefined ? options.shuffle : true;

    const n = X.length;
    const idx = Array.from({ length: n }, (_, i) => i);
    for (let epoch = 0; epoch < epochs; epoch++) {
      if (shuffle) {
        for (let i = n - 1; i > 0; i--) {
          const j = Math.floor(Math.random() * (i + 1));
          const tmp = idx[i]; idx[i] = idx[j]; idx[j] = tmp;
        }
      }
      for (let b = 0; b < n; b += batchSize) {
        const end = Math.min(b + batchSize, n);
        for (let i = b; i < end; i++) {
          const xi = X[idx[i]];
          const yi = y[idx[i]];
          this.trainSample(xi, yi, lr);
        }
      }
    }
  }

  accuracy(X, y) {
    if (!Array.isArray(X) || !Array.isArray(y) || X.length !== y.length) return NaN;
    let ok = 0;
    for (let i = 0; i < X.length; i++) if (this.predict(X[i]) === (y[i] ? 1 : 0)) ok++;
    return ok / X.length;
  }

  loss(X, y) {
    if (!Array.isArray(X) || !Array.isArray(y) || X.length !== y.length) return NaN;
    let total = 0;
    for (let i = 0; i < X.length; i++) {
      const p = Math.min(0.999999, Math.max(1e-6, this.forward(X[i])));
      const yi = y[i] ? 1 : 0;
      total -= yi * Math.log(p) + (1 - yi) * Math.log(1 - p);
    }
    return total / X.length;
  }

  reset(hidden = null) {
    if (hidden) this.hidden = Array.isArray(hidden) ? hidden : [hidden];
    this.initWeights();
  }
}
