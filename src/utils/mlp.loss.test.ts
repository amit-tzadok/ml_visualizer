import { describe, it, expect } from 'vitest';
import MLPClassifier from './mlp';

// Simple separable dataset (linear-ish)
const makeDataset = (n = 80) => {
  const X: number[][] = [];
  const y: number[] = [];
  for (let i = 0; i < n; i++) {
    const x1 = Math.random() * 2 - 1;
    const x2 = Math.random() * 2 - 1;
    const label = x1 * 0.4 + 0.1 > x2 ? 1 : 0;
    X.push([x1, x2]);
    y.push(label);
  }
  return { X, y };
};

describe('MLPClassifier loss history', () => {
  it('fit returns decreasing loss over epochs for separable data', () => {
    const { X, y } = makeDataset(100);
    const model = new MLPClassifier(2, [8], { lr: 0.3, optimizer: 'sgd' });
    const losses = model.fit(X, y, { epochs: 10, lr: 0.3, batchSize: 8, shuffle: true });
    expect(Array.isArray(losses)).toBe(true);
    expect(losses.length).toBe(10);
    // final loss should be <= initial (allow small numeric noise)
    expect(losses[losses.length - 1]).toBeLessThanOrEqual(losses[0] + 1e-6);
  });
});
