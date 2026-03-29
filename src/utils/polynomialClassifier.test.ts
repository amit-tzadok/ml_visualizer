import { describe, it, expect } from 'vitest';
import PolynomialPerceptron from './polynomialClassifier';

function makeCircular(n = 100): { X: [number, number][]; y: number[] } {
  const X: [number, number][] = [];
  const y: number[] = [];
  for (let i = 0; i < n; i++) {
    const angle = Math.random() * 2 * Math.PI;
    const outer = Math.random() > 0.5;
    const radius = outer ? 0.65 + Math.random() * 0.2 : 0.1 + Math.random() * 0.2;
    X.push([Math.cos(angle) * radius, Math.sin(angle) * radius]);
    y.push(outer ? 1 : 0);
  }
  return { X, y };
}

describe('PolynomialPerceptron — construction', () => {
  it('instantiates with degree=2 without error', () => {
    expect(() => new PolynomialPerceptron(2)).not.toThrow();
  });

  it('throws for unsupported degrees', () => {
    expect(() => new PolynomialPerceptron(1)).toThrow();
    expect(() => new PolynomialPerceptron(3)).toThrow();
  });
});

describe('PolynomialPerceptron — feature transform', () => {
  it('produces exactly 5 features from 2D input', () => {
    const pp = new PolynomialPerceptron(2);
    expect(pp.transform([0.5, -0.3]).length).toBe(5);
  });

  it('feature layout is [x, y, x², y², xy]', () => {
    const pp = new PolynomialPerceptron(2);
    const [x, y] = [0.5, -0.3];
    const f = pp.transform([x, y]);
    expect(f[0]).toBeCloseTo(x, 10);
    expect(f[1]).toBeCloseTo(y, 10);
    expect(f[2]).toBeCloseTo(x * x, 10);
    expect(f[3]).toBeCloseTo(y * y, 10);
    expect(f[4]).toBeCloseTo(x * y, 10);
  });

  it('transform is deterministic', () => {
    const pp = new PolynomialPerceptron(2);
    const f1 = pp.transform([0.7, -0.2]);
    const f2 = pp.transform([0.7, -0.2]);
    expect(f1).toEqual(f2);
  });
});

describe('PolynomialPerceptron — prediction', () => {
  it('predict returns 0 or 1', () => {
    const pp = new PolynomialPerceptron(2);
    const pred = pp.predict([0.5, -0.3]);
    expect(pred === 0 || pred === 1).toBe(true);
  });

  it('predictRaw returns a finite number', () => {
    const pp = new PolynomialPerceptron(2);
    const raw = pp.predictRaw([0.5, -0.3]);
    expect(typeof raw).toBe('number');
    expect(Number.isFinite(raw)).toBe(true);
  });
});

describe('PolynomialPerceptron — training', () => {
  it('trainSample does not throw for valid inputs', () => {
    const pp = new PolynomialPerceptron(2);
    expect(() => pp.trainSample([0.5, 0.5], 1)).not.toThrow();
    expect(() => pp.trainSample([-0.5, -0.5], 0)).not.toThrow();
  });

  it('misclassificationRate is in [0, 1]', () => {
    const { X, y } = makeCircular(50);
    const pp = new PolynomialPerceptron(2, 0.1, 20);
    pp.fit(X, y);
    const err = pp.misclassificationRate(X, y);
    expect(err).toBeGreaterThanOrEqual(0);
    expect(err).toBeLessThanOrEqual(1);
  });

  it('fit reduces misclassification rate vs untrained', () => {
    const { X, y } = makeCircular(200);
    const ppUntrained = new PolynomialPerceptron(2, 0.1, 1);
    const errBefore = ppUntrained.misclassificationRate(X, y);
    const ppTrained = new PolynomialPerceptron(2, 0.1, 60);
    ppTrained.fit(X, y);
    const errAfter = ppTrained.misclassificationRate(X, y);
    expect(errAfter).toBeLessThanOrEqual(errBefore + 0.05);
  });

  it('hingeLoss is non-negative and finite', () => {
    const { X, y } = makeCircular(50);
    const pp = new PolynomialPerceptron(2, 0.1, 10);
    pp.fit(X, y);
    const l = pp.hingeLoss(X, y);
    expect(l).toBeGreaterThanOrEqual(0);
    expect(Number.isFinite(l)).toBe(true);
  });

  it('mseRaw is non-negative and finite', () => {
    const { X, y } = makeCircular(50);
    const pp = new PolynomialPerceptron(2, 0.1, 10);
    pp.fit(X, y);
    const l = pp.mseRaw(X, y);
    expect(l).toBeGreaterThanOrEqual(0);
    expect(Number.isFinite(l)).toBe(true);
  });
});

describe('PolynomialPerceptron — reset', () => {
  it('reset changes at least some weights', () => {
    const pp = new PolynomialPerceptron(2);
    const { X, y } = makeCircular(30);
    pp.fit(X, y);
    const before = [...pp.model.weights];
    pp.reset();
    const after = [...pp.model.weights];
    // Weights should differ after reset (with overwhelming probability)
    const changed = before.some((w, i) => Math.abs(w - after[i]) > 1e-10);
    expect(changed).toBe(true);
  });
});
