import { describe, it, expect } from 'vitest';
import SVM from './svm';

// Linearly-separable XOR is NOT linearly separable, so we test on blobs.
function makeBlobs() {
  const X: number[][] = [];
  const y: number[] = [];
  for (let i = 0; i < 40; i++) {
    X.push([-0.6 + (Math.random() - 0.5) * 0.3, -0.4 + (Math.random() - 0.5) * 0.3]);
    y.push(0);
    X.push([0.6 + (Math.random() - 0.5) * 0.3, 0.4 + (Math.random() - 0.5) * 0.3]);
    y.push(1);
  }
  return { X, y };
}

describe('SVM', () => {
  it('predicts 0 or 1', () => {
    const svm = new SVM();
    expect([0, 1]).toContain(svm.predict([0.5, 0.5]));
    expect([0, 1]).toContain(svm.predict([-0.5, -0.5]));
  });

  it('predictProba is between 0 and 1', () => {
    const svm = new SVM();
    const p = svm.predictProba([1, 0]);
    expect(p).toBeGreaterThanOrEqual(0);
    expect(p).toBeLessThanOrEqual(1);
  });

  it('trains and achieves >85% accuracy on linearly separable blobs', () => {
    const { X, y } = makeBlobs();
    const svm = new SVM({ C: 2, lr: 0.02 });
    for (let i = 0; i < 50; i++) svm.fitStep(X, y);
    expect(svm.accuracy(X, y)).toBeGreaterThan(0.85);
  });

  it('decisionBoundary2D returns null when w1 ≈ 0', () => {
    const svm = new SVM();
    svm.weights = [1, 0];
    expect(svm.decisionBoundary2D()).toBeNull();
  });

  it('decisionBoundary2D returns valid slope/intercept', () => {
    const svm = new SVM();
    svm.weights = [1, 1];
    svm.bias = 0;
    const db = svm.decisionBoundary2D();
    expect(db).not.toBeNull();
    expect(db!.slope).toBeCloseTo(-1);
    expect(db!.intercept).toBeCloseTo(0);
  });

  it('marginLines2D returns two lines equidistant from boundary', () => {
    const svm = new SVM();
    svm.weights = [0, 1];
    svm.bias = 0;
    const ml = svm.marginLines2D();
    expect(ml).not.toBeNull();
    expect(ml!.pos.intercept).toBeCloseTo(1);
    expect(ml!.neg.intercept).toBeCloseTo(-1);
  });

  it('marginWidth decreases as weights grow', () => {
    const svm = new SVM();
    svm.weights = [1, 0];
    const w1 = svm.marginWidth();
    svm.weights = [2, 0];
    const w2 = svm.marginWidth();
    expect(w2).toBeLessThan(w1);
  });

  it('reset clears weights and bias', () => {
    const svm = new SVM();
    svm.weights = [3, 4];
    svm.bias = 1;
    svm.reset();
    expect(svm.weights).toEqual([0, 0]);
    expect(svm.bias).toBe(0);
  });
});
