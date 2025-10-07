
class Perceptron {
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
}
export default Perceptron;
