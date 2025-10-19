import Perceptron from "./perceptron";

// Simple polynomial feature wrapper around a Perceptron.
// Supports degree-2 polynomial features for 2D input: [x, y, x^2, y^2, x*y]
export default class PolynomialPerceptron {
  degree: number;
  model: Perceptron;

  constructor(degree = 2, learningRate = 0.05, epochs = 1) {
    if (degree !== 2) {
      throw new Error("Only degree=2 polynomial is supported currently");
    }
    this.degree = degree;
    this.model = new Perceptron(5, learningRate, epochs);
  }

  transform(v: [number, number]) {
    const [x, y] = v;
    return [x, y, x * x, y * y, x * y];
  }

  predict(rawV: [number, number]) {
    const features = this.transform(rawV);
    return this.model.predict(features as number[]);
  }

  trainSample(rawV: [number, number], label: number) {
    const features = this.transform(rawV);
    return this.model.trainSample(features as number[], label);
  }

  fit(Xraw: [number, number][], y: Array<number | string>) {
    const Xf = Xraw.map((v) => this.transform(v));
    return this.model.fit(Xf as number[][], y);
  }

  reset() {
    this.model.weights = Array.from({ length: this.model.weights.length }, () => Math.random() - 0.5);
    this.model.bias = Math.random() - 0.5;
  }

  misclassificationRate(Xraw: [number, number][], y: Array<number | string>) {
    if (!Array.isArray(Xraw) || !Array.isArray(y) || Xraw.length !== y.length) return NaN;
    const Xf = Xraw.map((v) => this.transform(v));
    return this.model.misclassificationRate(Xf as number[][], y);
  }

  hingeLoss(Xraw: [number, number][], y: Array<number | string>) {
    if (!Array.isArray(Xraw) || !Array.isArray(y) || Xraw.length !== y.length) return NaN;
    const Xf = Xraw.map((v) => this.transform(v));
    return this.model.hingeLoss(Xf as number[][], y);
  }

  mseRaw(Xraw: [number, number][], y: Array<number | string>) {
    if (!Array.isArray(Xraw) || !Array.isArray(y) || Xraw.length !== y.length) return NaN;
    const Xf = Xraw.map((v) => this.transform(v));
    return this.model.mseRaw(Xf as number[][], y);
  }

  fitHingeSGD(Xraw: [number, number][], y: Array<number | string>, options: any = {}) {
    if (!Array.isArray(Xraw) || !Array.isArray(y) || Xraw.length !== y.length) return;
    const Xf = Xraw.map((v) => this.transform(v));
    return this.model.fitHingeSGD(Xf as number[][], y, options);
  }
}
