// Lightweight vanilla MLP classifier (binary)
// Small, dependency-free implementation intended for interactive demos.

export type ActivationFunction = 'sigmoid' | 'tanh' | 'relu' | 'leaky_relu';

export class MLPClassifier {
  inputSize: number;
  hidden: number[];
  lr: number;
  optimizer: string;
  activation: ActivationFunction;
  momentumCoef: number;
  beta1: number;
  beta2: number;
  eps: number;
  iter: number;
  weights: number[][][];
  biases: number[][];
  // optimizer state
  velocityW: number[][][] | null;
  velocityB: number[][] | null;
  mW: number[][][] | null;
  vW: number[][][] | null;
  mB: number[][] | null;
  // safety caps to avoid numerical explosion in interactive demos
  private readonly _WEIGHT_CLAMP = 1e3;
  vB: number[][] | null;

  constructor(inputSize = 2, hidden: number[] | number = [16], options: { lr?: number; optimizer?: string; activation?: ActivationFunction; momentum?: number; beta1?: number; beta2?: number; eps?: number } = {}) {
    this.inputSize = inputSize;
    this.hidden = Array.isArray(hidden) ? hidden : [hidden];
  this.lr = options.lr || 0.3;
    this.optimizer = options.optimizer || "sgd";
    this.activation = options.activation || "sigmoid";
    this.momentumCoef = options.momentum ?? 0.9;
    this.beta1 = options.beta1 ?? 0.9;
    this.beta2 = options.beta2 ?? 0.999;
    this.eps = options.eps ?? 1e-8;
    this.iter = 0;
    this.velocityW = null;
    this.velocityB = null;
    this.mW = null;
    this.vW = null;
    this.mB = null;
    this.vB = null;
    this.initWeights();
  }

  initWeights() {
    this.weights = [];
    this.biases = [];
    let prev = this.inputSize;
    for (let i = 0; i < this.hidden.length; i++) {
      const h = this.hidden[i];
      // Xavier normal initialization: std = sqrt(2 / (fan_in + fan_out))
      const fan_in = prev;
      const fan_out = h;
      const std = Math.sqrt(2 / (fan_in + fan_out));
      const randn = () => {
        // Box-Muller transform
        let u = 0, v = 0;
        while (u === 0) u = Math.random();
        while (v === 0) v = Math.random();
        return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
      };
      this.weights.push(
        Array.from({ length: h }, () => Array.from({ length: prev }, () => randn() * std))
      );
      // initialize biases to zero (matches typical small-network setup)
      this.biases.push(Array.from({ length: h }, () => 0));
      prev = h;
    }
    // output layer: fan_out = 1
    const fan_in = prev;
    const fan_out = 1;
    const stdOut = Math.sqrt(2 / (fan_in + fan_out));
    const randnOut = () => {
      let u = 0, v = 0;
      while (u === 0) u = Math.random();
      while (v === 0) v = Math.random();
      return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    };
    this.weights.push(Array.from({ length: 1 }, () => Array.from({ length: prev }, () => randnOut() * stdOut)));
    this.biases.push([0]);

    // init optimizer state arrays
    this.initOptimizerState();
  }

  initOptimizerState() {
    // velocity for momentum (same shape as weights/biases)
    this.velocityW = this.weights.map((W) => W.map((row) => row.map(() => 0)));
    this.velocityB = this.biases.map((b) => b.map(() => 0));
    // adam m/v
    this.mW = this.weights.map((W) => W.map((row) => row.map(() => 0)));
    this.vW = this.weights.map((W) => W.map((row) => row.map(() => 0)));
    this.mB = this.biases.map((b) => b.map(() => 0));
    this.vB = this.biases.map((b) => b.map(() => 0));
    this.iter = 0;
  }

  static sigmoid(x: number) { return 1 / (1 + Math.exp(-x)); }
  static sigmoidDeriv(y: number) { return y * (1 - y); }
  
  static tanh(x: number) { return Math.tanh(x); }
  static tanhDeriv(y: number) { return 1 - y * y; }
  
  static relu(x: number) { return Math.max(0, x); }
  static reluDeriv(x: number) { return x > 0 ? 1 : 0; }
  
  static leakyRelu(x: number) { return x > 0 ? x : 0.01 * x; }
  static leakyReluDeriv(x: number) { return x > 0 ? 1 : 0.01; }
  
  private activate(x: number): number {
    switch (this.activation) {
      case 'tanh': return MLPClassifier.tanh(x);
      case 'relu': return MLPClassifier.relu(x);
      case 'leaky_relu': return MLPClassifier.leakyRelu(x);
      default: return MLPClassifier.sigmoid(x);
    }
  }
  
  private activateDeriv(y: number, z?: number): number {
    switch (this.activation) {
      case 'tanh': return MLPClassifier.tanhDeriv(y);
      case 'relu': return MLPClassifier.reluDeriv(z ?? y);
      case 'leaky_relu': return MLPClassifier.leakyReluDeriv(z ?? y);
      default: return MLPClassifier.sigmoidDeriv(y);
    }
  }

  forward(x: number[]) {
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
      if (l < this.weights.length - 1) {
        // Hidden layers use selected activation
        a = z.map((v) => this.activate(v));
      } else {
        // Output layer always uses sigmoid for binary classification
        a = z.map((v) => MLPClassifier.sigmoid(v));
      }
    }
    const out = a[0];
    // guard against non-finite outputs — return a neutral probability and log a message
    if (!Number.isFinite(out) || Number.isNaN(out)) {
      try { console.error('MLPClassifier.forward produced non-finite output', out); } catch (err) { void err; }
      return 0.5;
    }
    return out;
  }

  predict(x: number[]) {
    return this.forward(x) >= 0.5 ? 1 : 0;
  }

  trainSample(x: number[], y: number, lr: number | null = null) {
    const eta = lr == null ? this.lr : lr;
    // advance Adam time step once per sample (if using Adam)
    if (this.optimizer === "adam") this.iter += 1;
    const activations: number[][] = [x.slice()];
    let a = x.slice();
    const zs: number[][] = [];
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
      if (l < this.weights.length - 1) {
        // Hidden layers use selected activation
        a = z.map((v) => this.activate(v));
      } else {
        // Output layer uses sigmoid
        a = z.map((v) => MLPClassifier.sigmoid(v));
      }
      activations.push(a);
    }

  const pred = activations[activations.length - 1][0];
  if (!Number.isFinite(pred) || Number.isNaN(pred)) {
    try { console.error('MLPClassifier.trainSample: non-finite prediction, aborting sample update', pred); } catch (err) { void err; }
    return;
  }
  // For binary classification with sigmoid output and cross-entropy loss,
  // the output layer delta simplifies to (target - pred). Using the
  // sigmoid derivative here corresponds to an MSE-style gradient which
  // can slow or destabilize convergence. Use (y - pred) so gradient
  // direction matches the negative log-likelihood (cross-entropy).
  const delta_out = (y - pred);

    let delta: number[] = [delta_out];
    for (let l = this.weights.length - 1; l >= 0; l--) {
      const a_prev = activations[l];
      for (let j = 0; j < this.weights[l].length; j++) {
        const grad = delta[j];
        for (let k = 0; k < this.weights[l][j].length; k++) {
          // per-sample gradient for this weight
          let g = grad * a_prev[k];
          // gradient clipping to avoid exploding updates in interactive demos
          const CLIP = 5;
          if (g > CLIP) g = CLIP;
          else if (g < -CLIP) g = -CLIP;
          this.applyWeightUpdate(l, j, k, g, eta);
        }
        this.applyBiasUpdate(l, j, grad, eta);
      }
      if (l > 0) {
        const nextDelta = new Array(this.weights[l - 1].length).fill(0);
        for (let i = 0; i < this.weights[l].length; i++) {
          const row = this.weights[l][i];
          for (let k = 0; k < row.length; k++) {
            nextDelta[k] += row[k] * delta[i];
          }
        }
        // For ReLU/LeakyReLU we need the pre-activation values (z) for the derivative
        delta = nextDelta.map((v, idx) => {
          const z_val = zs[l - 1][idx];
          return v * this.activateDeriv(activations[l][idx], z_val);
        });
      }
    }
  }

  // internal param update with optimizer support
  private applyWeightUpdate(layer: number, j: number, k: number, grad: number, eta: number) {
    if (this.optimizer === "momentum") {
      if (!this.velocityW) this.initOptimizerState();
      const v = this.velocityW[layer][j][k] = (this.momentumCoef * this.velocityW[layer][j][k]) - eta * grad;
      this.weights[layer][j][k] += v;
    } else if (this.optimizer === "adam") {
      if (!this.mW || !this.vW) this.initOptimizerState();
      const m = (this.mW[layer][j][k] = this.beta1 * this.mW[layer][j][k] + (1 - this.beta1) * grad);
      const v = (this.vW[layer][j][k] = this.beta2 * this.vW[layer][j][k] + (1 - this.beta2) * grad * grad);
      const mHat = m / (1 - Math.pow(this.beta1, this.iter));
      const vHat = v / (1 - Math.pow(this.beta2, this.iter));
  // Keep same sign convention as SGD (which uses += eta * grad). Use += here
  // because `grad` passed to applyWeightUpdate is already (y - pred) * a_prev.
  this.weights[layer][j][k] += eta * (mHat / (Math.sqrt(vHat) + this.eps));
    } else {
      // sgd
      // Python uses additive updates: w += lr * grad (grad computed as (prev_activation * r))
      this.weights[layer][j][k] += eta * grad;
    }
    // Clamp weights to avoid numerical explosion in the browser
    try {
      let w = this.weights[layer][j][k];
      if (!Number.isFinite(w) || Number.isNaN(w)) w = 0;
      if (w > this._WEIGHT_CLAMP) w = this._WEIGHT_CLAMP;
      if (w < -this._WEIGHT_CLAMP) w = -this._WEIGHT_CLAMP;
      this.weights[layer][j][k] = w;
    } catch (err) {
      try { if (typeof console !== 'undefined' && console && console.debug) console.debug('mlp: weight clamp error', err); } catch { void 0; }
    }
  }

  private applyBiasUpdate(layer: number, j: number, grad: number, eta: number) {
    if (this.optimizer === "momentum") {
      if (!this.velocityB) this.initOptimizerState();
      const v = this.velocityB[layer][j] = (this.momentumCoef * this.velocityB[layer][j]) - eta * grad;
      this.biases[layer][j] += v;
    } else if (this.optimizer === "adam") {
      if (!this.mB || !this.vB) this.initOptimizerState();
      // iter already incremented in weight updates; if not, increment here
      const m = (this.mB[layer][j] = this.beta1 * this.mB[layer][j] + (1 - this.beta1) * grad);
      const v = (this.vB[layer][j] = this.beta2 * this.vB[layer][j] + (1 - this.beta2) * grad * grad);
      const mHat = m / (1 - Math.pow(this.beta1, this.iter));
      const vHat = v / (1 - Math.pow(this.beta2, this.iter));
  // Same sign fix for bias updates with Adam
  this.biases[layer][j] += eta * (mHat / (Math.sqrt(vHat) + this.eps));
    } else {
      this.biases[layer][j] += eta * grad;
    }
    // Clamp bias as well
    try {
      let b = this.biases[layer][j];
      if (!Number.isFinite(b) || Number.isNaN(b)) b = 0;
      if (b > this._WEIGHT_CLAMP) b = this._WEIGHT_CLAMP;
      if (b < -this._WEIGHT_CLAMP) b = -this._WEIGHT_CLAMP;
      this.biases[layer][j] = b;
    } catch (err) {
      try { if (typeof console !== 'undefined' && console && console.debug) console.debug('mlp: bias clamp error', err); } catch { void 0; }
    }
  }

  /**
   * Fit the model. Returns an array of epoch losses. Optional onEpoch callback receives (epochIndex, loss).
   */
  fit(
    X: number[][],
    y: number[],
    options: { epochs?: number; lr?: number; batchSize?: number; shuffle?: boolean; onEpoch?: (epoch: number, loss: number) => void } = {}
  ) {
    const epochs = options.epochs || 40;
    const lr = options.lr || this.lr;
    const batchSize = options.batchSize || 8;
    const shuffle = options.shuffle !== undefined ? options.shuffle : true;
    const onEpoch = options.onEpoch;

  const losses: number[] = [];
    const n = X.length;
    const idx = Array.from({ length: n }, (_, i) => i);
    for (let epoch = 0; epoch < epochs; epoch++) {
        try {
          // epoch body
        } catch (err) {
          try { console.error('MLPClassifier.fit: epoch error', err); } catch (e) { void e; }
          // push NaN to signal failure and break
          losses.push(NaN);
          break;
        }
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
      const L = this.loss(X, y);
      losses.push(L);
      if (typeof onEpoch === "function") {
        try { onEpoch(epoch, L); } catch (err) { if (typeof console !== 'undefined' && console && console.debug) console.debug('mlp: onEpoch callback error', err); }
      }
    }
    return losses;
  }

  accuracy(X: number[][], y: Array<number | string>) {
    if (!Array.isArray(X) || !Array.isArray(y) || X.length !== y.length) return NaN;
    let ok = 0;
    for (let i = 0; i < X.length; i++) if (this.predict(X[i]) === (y[i] ? 1 : 0)) ok++;
    return ok / X.length;
  }

  loss(X: number[][], y: Array<number | string>) {
    if (!Array.isArray(X) || !Array.isArray(y) || X.length !== y.length) return NaN;
    let total = 0;
    for (let i = 0; i < X.length; i++) {
      const p = Math.min(0.999999, Math.max(1e-6, this.forward(X[i])));
      const yi = y[i] ? 1 : 0;
      total -= yi * Math.log(p) + (1 - yi) * Math.log(1 - p);
    }
    return total / X.length;
  }

  reset(hidden: number[] | null = null) {
    if (hidden) this.hidden = Array.isArray(hidden) ? hidden : [hidden];
    this.iter = 0;
    this.initWeights();
  }
  
  setActivation(activation: ActivationFunction) {
    this.activation = activation;
    // Reinitialize weights when changing activation for better convergence
    this.initWeights();
  }
}

export default MLPClassifier;
