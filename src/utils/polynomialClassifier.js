import Perceptron from './perceptron.js';

// Simple polynomial feature wrapper around a Perceptron.
// Currently supports degree-2 polynomial features for 2D input: [x, y, x^2, y^2, x*y]
export default class PolynomialPerceptron {
  constructor(degree = 2, learningRate = 0.05, epochs = 1) {
    if (degree !== 2) {
      // for now we only support degree 2
      throw new Error('Only degree=2 polynomial is supported currently');
    }
    this.degree = degree;
    this.model = new Perceptron(5, learningRate, epochs);
  }

  // Transform raw 2D input into polynomial features [x, y, x^2, y^2, x*y]
  transform(v) {
    const [x, y] = v;
    return [x, y, x * x, y * y, x * y];
  }

  predict(rawV) {
    const features = this.transform(rawV);
    return this.model.predict(features);
  }

  // Train on a single raw sample
  trainSample(rawV, label) {
    const features = this.transform(rawV);
    return this.model.trainSample(features, label);
  }

  // Fit on arrays of raw samples
  fit(Xraw, y) {
    const Xf = Xraw.map((v) => this.transform(v));
    return this.model.fit(Xf, y);
  }

  // Reset weights/bias with small random values
  reset() {
    this.model.weights = Array.from({ length: this.model.weights.length }, () => Math.random() - 0.5);
    this.model.bias = Math.random() - 0.5;
  }

  // Evaluation helpers that operate on raw inputs
  misclassificationRate(Xraw, y) {
    if (!Array.isArray(Xraw) || !Array.isArray(y) || Xraw.length !== y.length)
      return NaN;
    const Xf = Xraw.map((v) => this.transform(v));
    return this.model.misclassificationRate(Xf, y);
  }

  hingeLoss(Xraw, y) {
    if (!Array.isArray(Xraw) || !Array.isArray(y) || Xraw.length !== y.length)
      return NaN;
    const Xf = Xraw.map((v) => this.transform(v));
    return this.model.hingeLoss(Xf, y);
  }

  mseRaw(Xraw, y) {
    if (!Array.isArray(Xraw) || !Array.isArray(y) || Xraw.length !== y.length)
      return NaN;
    const Xf = Xraw.map((v) => this.transform(v));
    return this.model.mseRaw(Xf, y);
  }

  // Hinge-SGD training wrapper (delegates to underlying perceptron)
  fitHingeSGD(Xraw, y, options = {}) {
    if (!Array.isArray(Xraw) || !Array.isArray(y) || Xraw.length !== y.length) return;
    const Xf = Xraw.map((v) => this.transform(v));
    return this.model.fitHingeSGD(Xf, y, options);
  }
}
