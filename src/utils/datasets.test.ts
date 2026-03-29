import { describe, it, expect } from 'vitest';
import {
  makeXOR,
  makeMoons,
  makeCircles,
  makeBlobs,
  makeSpirals,
  toXy,
  DataPoint,
} from './datasets';

function checkPoints(pts: DataPoint[], expectedN: number): void {
  expect(pts.length).toBe(expectedN);
  for (const p of pts) {
    expect(typeof p.x).toBe('number');
    expect(typeof p.y).toBe('number');
    expect(p.label === 0 || p.label === 1).toBe(true);
    expect(Number.isFinite(p.x)).toBe(true);
    expect(Number.isFinite(p.y)).toBe(true);
  }
}

function hasBothClasses(pts: DataPoint[]): boolean {
  const labels = new Set(pts.map((p) => p.label));
  return labels.has(0) && labels.has(1);
}

describe('makeXOR', () => {
  it('generates the requested number of points', () => {
    checkPoints(makeXOR(100), 100);
  });

  it('contains both classes', () => {
    expect(hasBothClasses(makeXOR(200, 0))).toBe(true);
  });

  it('with no noise, XOR labels match quadrant rule', () => {
    // With noise=0 (or very small) the label should match (x>0) === (y>0)
    const pts = makeXOR(400, 0);
    let correct = 0;
    for (const { x, y, label } of pts) {
      const expected: 0 | 1 = (x > 0) === (y > 0) ? 1 : 0;
      if (label === expected) correct++;
    }
    // Allow a tiny fraction for floating-point edge cases near axes
    expect(correct / pts.length).toBeGreaterThan(0.95);
  });

  it('default count is 200', () => {
    expect(makeXOR().length).toBe(200);
  });
});

describe('makeMoons', () => {
  it('generates n points', () => {
    checkPoints(makeMoons(80), 80);
  });

  it('contains both classes', () => {
    expect(hasBothClasses(makeMoons(200))).toBe(true);
  });

  it('class 0 and class 1 have roughly equal counts', () => {
    const pts = makeMoons(200);
    const n0 = pts.filter((p) => p.label === 0).length;
    const n1 = pts.filter((p) => p.label === 1).length;
    expect(Math.abs(n0 - n1)).toBeLessThanOrEqual(2);
  });

  it('default count is 200', () => {
    expect(makeMoons().length).toBe(200);
  });
});

describe('makeCircles', () => {
  it('generates n points', () => {
    checkPoints(makeCircles(120), 120);
  });

  it('contains both classes', () => {
    expect(hasBothClasses(makeCircles(200))).toBe(true);
  });

  it('inner ring (label=1) has smaller mean radius than outer ring (label=0)', () => {
    const pts = makeCircles(500, 0, 0.4);
    const outerR = pts.filter((p) => p.label === 0).map((p) => Math.hypot(p.x, p.y));
    const innerR = pts.filter((p) => p.label === 1).map((p) => Math.hypot(p.x, p.y));
    const meanOuter = outerR.reduce((s, v) => s + v, 0) / outerR.length;
    const meanInner = innerR.reduce((s, v) => s + v, 0) / innerR.length;
    expect(meanInner).toBeLessThan(meanOuter);
  });

  it('default count is 200', () => {
    expect(makeCircles().length).toBe(200);
  });
});

describe('makeBlobs', () => {
  it('generates n points', () => {
    checkPoints(makeBlobs(100), 100);
  });

  it('contains both classes', () => {
    expect(hasBothClasses(makeBlobs(200))).toBe(true);
  });

  it('class 0 centroid is left of class 1 centroid', () => {
    const pts = makeBlobs(1000, 0.02);
    const c0 = pts.filter((p) => p.label === 0);
    const c1 = pts.filter((p) => p.label === 1);
    const mean0x = c0.reduce((s, p) => s + p.x, 0) / c0.length;
    const mean1x = c1.reduce((s, p) => s + p.x, 0) / c1.length;
    expect(mean0x).toBeLessThan(mean1x);
  });

  it('default count is 200', () => {
    expect(makeBlobs().length).toBe(200);
  });
});

describe('makeSpirals', () => {
  it('generates n points', () => {
    checkPoints(makeSpirals(100), 100);
  });

  it('contains both classes', () => {
    expect(hasBothClasses(makeSpirals(200))).toBe(true);
  });

  it('roughly balanced classes', () => {
    const pts = makeSpirals(200);
    const n0 = pts.filter((p) => p.label === 0).length;
    const n1 = pts.filter((p) => p.label === 1).length;
    expect(Math.abs(n0 - n1)).toBeLessThanOrEqual(2);
  });

  it('default count is 200', () => {
    expect(makeSpirals().length).toBe(200);
  });
});

describe('toXy', () => {
  it('converts DataPoint array to {X, y}', () => {
    const pts: DataPoint[] = [
      { x: 1, y: 2, label: 0 },
      { x: 3, y: 4, label: 1 },
    ];
    const { X, y } = toXy(pts);
    expect(X).toEqual([[1, 2], [3, 4]]);
    expect(y).toEqual([0, 1]);
  });

  it('handles empty array', () => {
    const { X, y } = toXy([]);
    expect(X).toEqual([]);
    expect(y).toEqual([]);
  });

  it('X has 2 features per sample', () => {
    const pts = makeBlobs(50);
    const { X } = toXy(pts);
    for (const row of X) expect(row.length).toBe(2);
  });

  it('y values are exactly 0 or 1', () => {
    const pts = makeXOR(100);
    const { y } = toXy(pts);
    for (const label of y) expect(label === 0 || label === 1).toBe(true);
  });
});
