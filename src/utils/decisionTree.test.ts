import { describe, it, expect } from 'vitest';
import DecisionTree from './decisionTree';

function makeSeparable(n = 100): { X: number[][]; y: number[] } {
  const X: number[][] = [];
  const y: number[] = [];
  for (let i = 0; i < n; i++) {
    const x = Math.random() * 2 - 1;
    const yv = Math.random() * 2 - 1;
    X.push([x, yv]);
    y.push(x > 0 ? 1 : 0);  // vertical split at x=0
  }
  return { X, y };
}

function makeXOR(n = 120): { X: number[][]; y: number[] } {
  const X: number[][] = [];
  const y: number[] = [];
  for (let i = 0; i < n; i++) {
    const x = (Math.random() * 2 - 1) * 0.9;
    const yv = (Math.random() * 2 - 1) * 0.9;
    X.push([x, yv]);
    y.push((x > 0) === (yv > 0) ? 1 : 0);
  }
  return { X, y };
}

describe('DecisionTree — basic behaviour', () => {
  it('returns 0 (default) before fitting', () => {
    const dt = new DecisionTree();
    expect(dt.predictSample([0, 0])).toBe(0);
  });

  it('predict returns array of same length as input', () => {
    const { X, y } = makeSeparable(30);
    const dt = new DecisionTree({ maxDepth: 3 });
    dt.fit(X, y);
    const preds = dt.predict(X);
    expect(preds.length).toBe(X.length);
  });

  it('all predictions are 0 or 1', () => {
    const { X, y } = makeSeparable(40);
    const dt = new DecisionTree({ maxDepth: 3 });
    dt.fit(X, y);
    for (const p of dt.predict(X)) expect(p === 0 || p === 1).toBe(true);
  });

  it('accuracy is in [0, 1]', () => {
    const { X, y } = makeSeparable(50);
    const dt = new DecisionTree({ maxDepth: 3 });
    dt.fit(X, y);
    const acc = dt.accuracy(X, y);
    expect(acc).toBeGreaterThanOrEqual(0);
    expect(acc).toBeLessThanOrEqual(1);
  });

  it('accuracy of empty dataset is 0', () => {
    const dt = new DecisionTree();
    expect(dt.accuracy([], [])).toBe(0);
  });
});

describe('DecisionTree — accuracy', () => {
  it('achieves >90% accuracy on clean linearly separable data', () => {
    const { X, y } = makeSeparable(120);
    const dt = new DecisionTree({ maxDepth: 5, minSamplesSplit: 2, minSamplesLeaf: 1 });
    dt.fit(X, y);
    expect(dt.accuracy(X, y)).toBeGreaterThan(0.9);
  });

  it('handles XOR with sufficient depth', () => {
    const { X, y } = makeXOR(200);
    const dt = new DecisionTree({ maxDepth: 6, minSamplesSplit: 4, minSamplesLeaf: 2 });
    dt.fit(X, y);
    expect(dt.accuracy(X, y)).toBeGreaterThan(0.82);
  });

  it('deeper tree has equal or better training accuracy than shallower', () => {
    const { X, y } = makeXOR(150);
    const dtShallow = new DecisionTree({ maxDepth: 1, minSamplesSplit: 4, minSamplesLeaf: 2 });
    const dtDeep = new DecisionTree({ maxDepth: 6, minSamplesSplit: 4, minSamplesLeaf: 2 });
    dtShallow.fit(X, y);
    dtDeep.fit(X, y);
    expect(dtDeep.accuracy(X, y)).toBeGreaterThanOrEqual(dtShallow.accuracy(X, y) - 0.01);
  });
});

describe('DecisionTree — edge cases', () => {
  it('maxDepth=0 forces a single leaf predicting majority class', () => {
    const X = [[0, 0], [1, 0], [0, 1], [1, 1]];
    const y = [1, 1, 1, 0];  // majority = 1
    const dt = new DecisionTree({ maxDepth: 0, minSamplesSplit: 2, minSamplesLeaf: 1 });
    dt.fit(X, y);
    expect(dt.predictSample([0.5, 0.5])).toBe(1);
  });

  it('single-class dataset predicts that class everywhere', () => {
    const X = [[0, 0], [1, 0], [-1, 0]];
    const y = [0, 0, 0];
    const dt = new DecisionTree();
    dt.fit(X, y);
    expect(dt.predictSample([5, 5])).toBe(0);
    expect(dt.predictSample([-5, -5])).toBe(0);
  });

  it('fit on empty dataset does not throw', () => {
    const dt = new DecisionTree();
    expect(() => dt.fit([], [])).not.toThrow();
    expect(dt.root).toBeNull();
  });
});

describe('DecisionTree — split boundaries', () => {
  it('getSplitBoundaries returns non-empty array for fitted tree', () => {
    const { X, y } = makeSeparable(60);
    const dt = new DecisionTree({ maxDepth: 3 });
    dt.fit(X, y);
    const bounds = dt.getSplitBoundaries();
    expect(Array.isArray(bounds)).toBe(true);
    expect(bounds.length).toBeGreaterThan(0);
  });

  it('deeper tree has more boundaries than shallower', () => {
    const { X, y } = makeXOR(200);
    const dtShallow = new DecisionTree({ maxDepth: 1, minSamplesSplit: 4, minSamplesLeaf: 2 });
    const dtDeep = new DecisionTree({ maxDepth: 4, minSamplesSplit: 4, minSamplesLeaf: 2 });
    dtShallow.fit(X, y);
    dtDeep.fit(X, y);
    expect(dtDeep.getSplitBoundaries().length).toBeGreaterThanOrEqual(
      dtShallow.getSplitBoundaries().length
    );
  });

  it('returns empty array before fitting', () => {
    const dt = new DecisionTree();
    expect(dt.getSplitBoundaries()).toEqual([]);
  });

  it('boundary depth values are non-negative integers', () => {
    const { X, y } = makeSeparable(50);
    const dt = new DecisionTree({ maxDepth: 4 });
    dt.fit(X, y);
    for (const b of dt.getSplitBoundaries()) {
      expect(b.depth).toBeGreaterThanOrEqual(0);
      expect(Number.isInteger(b.depth)).toBe(true);
    }
  });
});

describe('DecisionTree — treeDepth and reset', () => {
  it('treeDepth is 0 before fit', () => {
    const dt = new DecisionTree();
    expect(dt.treeDepth()).toBe(0);
  });

  it('treeDepth is >= 1 after fit on non-trivial data', () => {
    const { X, y } = makeSeparable(40);
    const dt = new DecisionTree({ maxDepth: 3 });
    dt.fit(X, y);
    expect(dt.treeDepth()).toBeGreaterThanOrEqual(1);
  });

  it('treeDepth does not exceed maxDepth + 1', () => {
    const { X, y } = makeXOR(200);
    const maxD = 3;
    const dt = new DecisionTree({ maxDepth: maxD });
    dt.fit(X, y);
    expect(dt.treeDepth()).toBeLessThanOrEqual(maxD + 1);
  });

  it('reset clears the tree', () => {
    const { X, y } = makeSeparable(40);
    const dt = new DecisionTree({ maxDepth: 3 });
    dt.fit(X, y);
    expect(dt.root).not.toBeNull();
    dt.reset();
    expect(dt.root).toBeNull();
    expect(dt.treeDepth()).toBe(0);
  });
});
