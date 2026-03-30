import { describe, it, expect } from 'vitest';
import RandomForest from './randomForest';

function makeBlobs() {
  const X: number[][] = [];
  const y: number[] = [];
  for (let i = 0; i < 60; i++) {
    X.push([-0.6 + (Math.random() - 0.5) * 0.3, -0.4 + (Math.random() - 0.5) * 0.3]);
    y.push(0);
    X.push([0.6 + (Math.random() - 0.5) * 0.3, 0.4 + (Math.random() - 0.5) * 0.3]);
    y.push(1);
  }
  return { X, y };
}

describe('RandomForest', () => {
  it('predicts 0 before fit', () => {
    const rf = new RandomForest();
    expect(rf.predictSample([0, 0])).toBe(0);
  });

  it('predictProba is 0.5 before fit', () => {
    const rf = new RandomForest();
    expect(rf.predictProba([0, 0])).toBe(0.5);
  });

  it('fit creates the correct number of trees', () => {
    const { X, y } = makeBlobs();
    const rf = new RandomForest({ nEstimators: 10 });
    rf.fit(X, y);
    expect(rf.trees.length).toBe(10);
  });

  it('achieves >90% accuracy on linearly separable blobs', () => {
    const { X, y } = makeBlobs();
    const rf = new RandomForest({ nEstimators: 15, maxDepth: 4 });
    rf.fit(X, y);
    expect(rf.accuracy(X, y)).toBeGreaterThan(0.9);
  });

  it('predictProba is between 0 and 1', () => {
    const { X, y } = makeBlobs();
    const rf = new RandomForest({ nEstimators: 5 });
    rf.fit(X, y);
    const p = rf.predictProba([0, 0]);
    expect(p).toBeGreaterThanOrEqual(0);
    expect(p).toBeLessThanOrEqual(1);
  });

  it('addTree increments tree count', () => {
    const { X, y } = makeBlobs();
    const rf = new RandomForest({ nEstimators: 5 });
    rf.fit(X, y);
    expect(rf.trees.length).toBe(5);
    rf.addTree(X, y);
    expect(rf.trees.length).toBe(6);
  });

  it('reset clears trees', () => {
    const { X, y } = makeBlobs();
    const rf = new RandomForest({ nEstimators: 5 });
    rf.fit(X, y);
    rf.reset();
    expect(rf.trees.length).toBe(0);
  });

  it('predict returns array of 0/1', () => {
    const { X, y } = makeBlobs();
    const rf = new RandomForest({ nEstimators: 5 });
    rf.fit(X, y);
    const preds = rf.predict(X);
    expect(preds.length).toBe(X.length);
    for (const p of preds) expect([0, 1]).toContain(p);
  });
});
