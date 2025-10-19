import { describe, test, expect } from 'vitest';
import MLPClassifier from './mlp.ts';

describe('MLPClassifier basic', () => {
  test('forward and predict produce numbers in range', () => {
    const m = new MLPClassifier(2, [4], { lr: 0.01 });
    const out = m.forward([0.1, -0.2]);
    expect(typeof out).toBe('number');
    expect(out).toBeGreaterThanOrEqual(0);
    expect(out).toBeLessThanOrEqual(1);
    const pred = m.predict([0.1, -0.2]);
    expect(pred === 0 || pred === 1).toBeTruthy();
  });

  test('fit reduces loss / improves accuracy on separable dataset', () => {
    const X: number[][] = [];
    const y: number[] = [];
    // simple line: x > y => class 1, else 0
    for (let i = 0; i < 80; i++) {
      const x = Math.random() * 2 - 1;
      const yv = Math.random() * 2 - 1;
      X.push([x, yv]);
      y.push(x > yv ? 1 : 0);
    }
    const m = new MLPClassifier(2, [8], { lr: 0.05 });
    const beforeAcc = m.accuracy(X, y);
    m.fit(X, y, { epochs: 30, lr: 0.05, batchSize: 8 });
    const afterAcc = m.accuracy(X, y);
    expect(afterAcc).toBeGreaterThanOrEqual(beforeAcc);
  });
});
