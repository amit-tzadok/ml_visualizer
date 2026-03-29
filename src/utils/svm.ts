/**
 * Support Vector Machine with SGD on hinge loss.
 * Supports linear and RBF kernels.
 *
 * Linear:  decision boundary w·x + b = 0, margin = 2 / ‖w‖
 * RBF:     approximated via Random Fourier Features (Rahimi & Recht 2007)
 *          K(x,z) ≈ exp(−γ‖x−z‖²)  using D random frequency components
 */

export interface SVMOptions {
  C?:           number;            // soft-margin penalty      (default 1.0)
  lr?:          number;            // learning rate            (default 0.01)
  kernel?:      "linear" | "rbf"; // kernel type              (default "linear")
  gamma?:       number;            // RBF bandwidth            (default 1.0)
  nComponents?: number;            // RFF dimensions for RBF   (default 64)
}

/** Box-Muller standard-normal sample. */
function gaussianRandom(): number {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

export class SVM {
  weights:    number[];
  bias:       number;
  C:          number;
  lr:         number;
  kernel:     "linear" | "rbf";
  gamma:      number;
  readonly nComponents: number;

  private _omegas: number[][];   // D×2 random frequencies for RFF
  private _biases: number[];     // D random phase shifts for RFF

  constructor(options: SVMOptions = {}) {
    this.C           = options.C           ?? 1.0;
    this.lr          = options.lr          ?? 0.01;
    this.kernel      = options.kernel      ?? "linear";
    this.gamma       = options.gamma       ?? 1.0;
    this.nComponents = options.nComponents ?? 64;
    this._omegas = [];
    this._biases = [];
    this.bias    = 0;
    this.weights = [];
    this._initKernel();
  }

  private _initKernel(): void {
    if (this.kernel === "rbf") {
      this._sampleRFF();
      this.weights = new Array(this.nComponents).fill(0);
    } else {
      this._omegas = [];
      this._biases = [];
      this.weights = [0, 0];
    }
    this.bias = 0;
  }

  private _sampleRFF(): void {
    const D     = this.nComponents;
    const scale = Math.sqrt(2 * this.gamma);
    this._omegas = Array.from({ length: D }, () => [
      gaussianRandom() * scale,
      gaussianRandom() * scale,
    ]);
    this._biases = Array.from({ length: D }, () => Math.random() * 2 * Math.PI);
  }

  /** Switch kernel — resets weights and re-samples RFF when switching to rbf. */
  setKernel(kernel: "linear" | "rbf", gamma?: number): void {
    this.kernel = kernel;
    if (gamma !== undefined) this.gamma = gamma;
    this._initKernel();
  }

  private _transform(x: number[]): number[] {
    if (this.kernel === "linear") return x;
    const D     = this.nComponents;
    const scale = Math.sqrt(2 / D);
    return this._omegas.map((omega, i) =>
      scale * Math.cos(omega[0] * x[0] + omega[1] * x[1] + this._biases[i])
    );
  }

  private _rawScore(x: number[]): number {
    const phi = this._transform(x);
    let s = this.bias;
    for (let i = 0; i < phi.length; i++) s += this.weights[i] * phi[i];
    return s;
  }

  /** One SGD epoch. Returns average hinge loss. */
  fitStep(X: number[][], y: number[]): number {
    const n = X.length;
    if (!n) return 0;
    const D = this.weights.length;

    // Fisher-Yates shuffle
    const idx = Array.from({ length: n }, (_, i) => i);
    for (let i = n - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [idx[i], idx[j]] = [idx[j], idx[i]];
    }

    let totalLoss = 0;
    for (const i of idx) {
      const phi = this._transform(X[i]);
      const yi  = y[i] === 1 ? 1 : -1;
      let raw = this.bias;
      for (let d = 0; d < D; d++) raw += this.weights[d] * phi[d];
      const margin = yi * raw;

      if (margin < 1) {
        // In margin or misclassified — update
        for (let d = 0; d < D; d++) {
          this.weights[d] += this.lr * (this.C * yi * phi[d] - this.weights[d]);
        }
        this.bias += this.lr * this.C * yi;
        totalLoss += 1 - margin;
      } else {
        // Outside margin — regularise only
        for (let d = 0; d < D; d++) {
          this.weights[d] -= this.lr * this.weights[d];
        }
      }
    }
    return totalLoss / n;
  }

  predict(x: number[]): number {
    return this._rawScore(x) >= 0 ? 1 : 0;
  }

  /** Soft probability via sigmoid — used for heatmap rendering. */
  predictProba(x: number[]): number {
    const s = this._rawScore(x);
    return 1 / (1 + Math.exp(-s));
  }

  /** Geometric margin width = 2 / ‖w‖ (linear only). */
  marginWidth(): number {
    if (this.kernel !== "linear") return NaN;
    const norm = Math.sqrt(this.weights[0] ** 2 + this.weights[1] ** 2);
    return norm < 1e-10 ? Infinity : 2 / norm;
  }

  /** Decision boundary (w·x+b=0) as slope-intercept (linear only). */
  decisionBoundary2D(): { slope: number; intercept: number } | null {
    if (this.kernel !== "linear") return null;
    const [w0, w1] = this.weights;
    if (Math.abs(w1) < 1e-10) return null;
    return { slope: -w0 / w1, intercept: -this.bias / w1 };
  }

  /** Margin lines (w·x+b=±1) as slope-intercept (linear only). */
  marginLines2D(): {
    pos: { slope: number; intercept: number };
    neg: { slope: number; intercept: number };
  } | null {
    if (this.kernel !== "linear") return null;
    const [w0, w1] = this.weights;
    if (Math.abs(w1) < 1e-10) return null;
    return {
      pos: { slope: -w0 / w1, intercept: (-this.bias + 1) / w1 },
      neg: { slope: -w0 / w1, intercept: (-this.bias - 1) / w1 },
    };
  }

  /** True for points on/inside the margin band (linear only). */
  isSupportVector(x: number[], label: number, epsilon = 0.2): boolean {
    if (this.kernel !== "linear") return false;
    const yi = label === 1 ? 1 : -1;
    return yi * this._rawScore(x) <= 1 + epsilon;
  }

  accuracy(X: number[][], y: number[]): number {
    if (!X.length) return 0;
    return X.filter((x, i) => this.predict(x) === y[i]).length / X.length;
  }

  reset(): void {
    this.weights = this.weights.map(() => 0);
    this.bias = 0;
  }
}

export default SVM;
