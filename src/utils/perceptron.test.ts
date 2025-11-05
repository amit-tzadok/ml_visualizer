import { describe, it, expect } from 'vitest';
import Perceptron from './perceptron';

function makeSeparable(n = 100) {
  const X: number[][] = [];
  const y: number[] = [];
  for (let i = 0; i < n; i++) {
    const x = Math.random() * 2 - 1;
    const yv = Math.random() * 2 - 1;
    const label = x * 0.8 + 0.1 > yv ? 1 : 0;
    X.push([x, yv]);
    y.push(label);
  }
  return { X, y };
}

function makeNoisy(n = 150, noise = 0.15) {
  const X: number[][] = [];
  const y: number[] = [];
  for (let i = 0; i < n; i++) {
    const x = Math.random() * 2 - 1;
    const yv = Math.random() * 2 - 1;
    let label = x * 0.3 + 0.0 > yv ? 1 : 0;
    if (Math.random() < noise) label = label ? 0 : 1;
    X.push([x, yv]);
    y.push(label);
  }
  return { X, y };
}

describe('Perceptron advanced training', () => {
  it('converges on linearly separable data with fitOnline', () => {
    const { X, y } = makeSeparable(120);
    const p = new Perceptron(2, 0.05, 50);
    p.fitOnline(X, y, { epochs: 50, shuffle: true, earlyStop: { patience: 3, metric: 'errors' } });
    const err = p.misclassificationRate(X, y);
    expect(err).toBeLessThan(0.05);
  });

  it('averaged perceptron produces low error on separable data', () => {
    const { X, y } = makeSeparable(120);
    const p = new Perceptron(2, 0.05, 20);
    p.fitAveraged(X, y, { epochs: 20, shuffle: true });
    const err = p.misclassificationRate(X, y);
    expect(err).toBeLessThan(0.08);
  });

  it('pocket algorithm handles noisy data better than naive one pass', () => {
    const { X, y } = makeNoisy(200, 0.2);
    const base = new Perceptron(2, 0.05, 1);
    base.fit(X, y);
    const baseErr = base.misclassificationRate(X, y);

    const p = new Perceptron(2, 0.05, 10);
    p.fitPocket(X, y, { epochs: 10, shuffle: true });
    const err = p.misclassificationRate(X, y);

    expect(err).toBeLessThanOrEqual(baseErr);
  });
});
