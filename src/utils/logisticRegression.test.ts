import { describe, it, expect } from 'vitest';
import LogisticRegression from './logisticRegression';

function makeSeparable(n = 100): { X: number[][]; y: number[] } {
  const X: number[][] = [];
  const y: number[] = [];
  for (let i = 0; i < n; i++) {
    const x = Math.random() * 2 - 1;
    const yv = Math.random() * 2 - 1;
    X.push([x, yv]);
    y.push(x + 0.3 > yv ? 1 : 0);
  }
  return { X, y };
}

describe('LogisticRegression — core behaviour', () => {
  it('probability is in [0, 1]', () => {
    const lr = new LogisticRegression(2);
    const p = lr.probability([0.5, -0.3]);
    expect(p).toBeGreaterThanOrEqual(0);
    expect(p).toBeLessThanOrEqual(1);
  });

  it('predict returns 0 or 1', () => {
    const lr = new LogisticRegression(2);
    const pred = lr.predict([0.5, -0.3]);
    expect(pred === 0 || pred === 1).toBe(true);
  });

  it('sigmoid is numerically stable for extreme inputs', () => {
    expect(LogisticRegression.sigmoid(1000)).toBeCloseTo(1, 5);
    expect(LogisticRegression.sigmoid(-1000)).toBeCloseTo(0, 5);
    expect(LogisticRegression.sigmoid(0)).toBeCloseTo(0.5, 5);
  });

  it('loss is non-negative and finite', () => {
    const { X, y } = makeSeparable(50);
    const lr = new LogisticRegression(2);
    const l = lr.loss(X, y);
    expect(l).toBeGreaterThanOrEqual(0);
    expect(Number.isFinite(l)).toBe(true);
  });

  it('accuracy returns value in [0, 1]', () => {
    const { X, y } = makeSeparable(50);
    const lr = new LogisticRegression(2);
    const acc = lr.accuracy(X, y);
    expect(acc).toBeGreaterThanOrEqual(0);
    expect(acc).toBeLessThanOrEqual(1);
  });

  it('loss of empty dataset is 0', () => {
    const lr = new LogisticRegression(2);
    expect(lr.loss([], [])).toBe(0);
  });

  it('accuracy of empty dataset is 0', () => {
    const lr = new LogisticRegression(2);
    expect(lr.accuracy([], [])).toBe(0);
  });
});

describe('LogisticRegression — training', () => {
  it('fit returns loss array', () => {
    const { X, y } = makeSeparable(60);
    const lr = new LogisticRegression(2, { lr: 0.3 });
    const losses = lr.fit(X, y, { epochs: 20, lr: 0.3 });
    expect(Array.isArray(losses)).toBe(true);
    expect(losses.length).toBeGreaterThan(0);
    for (const l of losses) expect(Number.isFinite(l)).toBe(true);
  });

  it('loss decreases over training on separable data', () => {
    const { X, y } = makeSeparable(150);
    const lr = new LogisticRegression(2, { lr: 0.5, lambda: 0 });
    const losses = lr.fit(X, y, { epochs: 80 });
    expect(losses[losses.length - 1]).toBeLessThanOrEqual(losses[0] + 1e-4);
  });

  it('achieves > 85% accuracy on linearly separable data', () => {
    const { X, y } = makeSeparable(200);
    const lr = new LogisticRegression(2, { lr: 0.5, lambda: 0 });
    lr.fit(X, y, { epochs: 150 });
    expect(lr.accuracy(X, y)).toBeGreaterThan(0.85);
  });

  it('fitStep returns finite loss', () => {
    const { X, y } = makeSeparable(50);
    const lr = new LogisticRegression(2);
    const l = lr.fitStep(X, y, 0.1);
    expect(Number.isFinite(l)).toBe(true);
    expect(l).toBeGreaterThanOrEqual(0);
  });

  it('fitStep returns Infinity for empty dataset', () => {
    const lr = new LogisticRegression(2);
    const l = lr.fitStep([], []);
    expect(l).toBe(Infinity);
  });

  it('trainHistory is populated after fit', () => {
    const { X, y } = makeSeparable(40);
    const lr = new LogisticRegression(2);
    lr.fit(X, y, { epochs: 5 });
    expect(lr.trainHistory.length).toBeGreaterThan(0);
    for (const r of lr.trainHistory) {
      expect(typeof r.epoch).toBe('number');
      expect(typeof r.loss).toBe('number');
      expect(typeof r.accuracy).toBe('number');
    }
  });

  it('onEpoch callback is called each epoch', () => {
    const { X, y } = makeSeparable(40);
    const lr = new LogisticRegression(2);
    const epochsCalled: number[] = [];
    lr.fit(X, y, { epochs: 5, onEpoch: (e) => epochsCalled.push(e) });
    expect(epochsCalled.length).toBe(5);
    expect(epochsCalled[0]).toBe(0);
  });
});

describe('LogisticRegression — utilities', () => {
  it('decisionBoundary2D returns slope and intercept for 2D weights', () => {
    const lr = new LogisticRegression(2);
    const db = lr.decisionBoundary2D();
    expect(db).not.toBeNull();
    expect(typeof db!.slope).toBe('number');
    expect(typeof db!.intercept).toBe('number');
  });

  it('decisionBoundary2D returns null for non-2D weights', () => {
    const lr = new LogisticRegression(3);
    expect(lr.decisionBoundary2D()).toBeNull();
  });

  it('reset clears trainHistory', () => {
    const lr = new LogisticRegression(2);
    const { X, y } = makeSeparable(30);
    lr.fit(X, y, { epochs: 5 });
    expect(lr.trainHistory.length).toBeGreaterThan(0);
    lr.reset();
    expect(lr.trainHistory.length).toBe(0);
  });

  it('reset reinitializes bias to 0', () => {
    const lr = new LogisticRegression(2);
    const { X, y } = makeSeparable(30);
    lr.fit(X, y, { epochs: 10 });
    lr.reset();
    expect(lr.bias).toBe(0);
  });

  it('L2 regularization keeps weights bounded', () => {
    const { X, y } = makeSeparable(100);
    const lr = new LogisticRegression(2, { lr: 1.0, lambda: 0.5 });
    lr.fit(X, y, { epochs: 200 });
    for (const w of lr.weights) {
      expect(Math.abs(w)).toBeLessThan(100);
    }
  });
});
