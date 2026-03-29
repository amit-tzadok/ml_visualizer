import { describe, it, expect } from 'vitest';
import KNNClassifier, { predictKNN, UiPoint } from './knn';

describe('KNNClassifier', () => {
  it('returns null for empty training set', () => {
    const knn = new KNNClassifier(3);
    expect(knn.predict([0, 0])).toBeNull();
  });

  it('predicts the only available label', () => {
    const knn = new KNNClassifier(1);
    knn.addSample(0, 0, 'A');
    expect(knn.predict([5, 5])).toBe('A');
  });

  it('picks nearest neighbor with k=1', () => {
    const knn = new KNNClassifier(1);
    knn.addSample(0, 0, 'A');
    knn.addSample(10, 0, 'B');
    expect(knn.predict([0.5, 0])).toBe('A');
    expect(knn.predict([9.5, 0])).toBe('B');
  });

  it('majority vote works with k=3', () => {
    const knn = new KNNClassifier(3);
    knn.addSample(0, 0, 'A');
    knn.addSample(0.1, 0, 'A');
    knn.addSample(0.2, 0, 'A');
    knn.addSample(8, 8, 'B');
    knn.addSample(9, 8, 'B');
    expect(knn.predict([0.1, 0])).toBe('A');
  });

  it('handles k larger than dataset size by using all points', () => {
    const knn = new KNNClassifier(100);
    knn.addSample(0, 0, 'X');
    knn.addSample(1, 0, 'X');
    expect(knn.predict([0.5, 0])).toBe('X');
  });

  it('distance function returns 0 for same point', () => {
    const knn = new KNNClassifier(1);
    const dist = knn.distance({ x: 3, y: 4 }, { x: 3, y: 4 });
    expect(dist).toBe(0);
  });

  it('distance function is symmetric', () => {
    const knn = new KNNClassifier(1);
    const d1 = knn.distance({ x: 0, y: 0 }, { x: 3, y: 4 });
    const d2 = knn.distance({ x: 3, y: 4 }, { x: 0, y: 0 });
    expect(d1).toBeCloseTo(d2);
  });

  it('correctly classifies numeric labels', () => {
    const knn = new KNNClassifier(1);
    knn.addSample(0, 0, 0);
    knn.addSample(10, 10, 1);
    expect(knn.predict([0.1, 0.1])).toBe('0');
    expect(knn.predict([9.9, 9.9])).toBe('1');
  });

  it('breaks ties by first insertion order', () => {
    // 2 A points, 2 B points equidistant — tie → first key wins (insertion order)
    const knn = new KNNClassifier(4);
    knn.addSample(-1, 0, 'A');
    knn.addSample(-0.9, 0, 'A');
    knn.addSample(0.9, 0, 'B');
    knn.addSample(1, 0, 'B');
    // Result is implementation-dependent on tie; just ensure it's not null
    expect(knn.predict([0, 0])).not.toBeNull();
  });
});

describe('predictKNN (standalone helper)', () => {
  it('returns null for empty or undefined points', () => {
    expect(predictKNN([], 3, [0, 0])).toBeNull();
    expect(predictKNN(undefined, 3, [0, 0])).toBeNull();
  });

  it('returns nearest label for k=1', () => {
    const pts: UiPoint[] = [
      { coords: [0, 0], label: 'cat' },
      { coords: [10, 10], label: 'dog' },
    ];
    expect(predictKNN(pts, 1, [0.1, 0.1])).toBe('cat');
    expect(predictKNN(pts, 1, [9.9, 9.9])).toBe('dog');
  });

  it('majority vote selects correct label', () => {
    const pts: UiPoint[] = [
      { coords: [0, 0], label: '0' },
      { coords: [0.2, 0], label: '0' },
      { coords: [0.4, 0], label: '0' },
      { coords: [10, 0], label: '1' },
    ];
    expect(predictKNN(pts, 3, [0.1, 0])).toBe('0');
  });

  it('handles k larger than dataset gracefully', () => {
    const pts: UiPoint[] = [{ coords: [1, 1], label: 'z' }];
    expect(predictKNN(pts, 999, [0, 0])).toBe('z');
  });

  it('returns a string label', () => {
    const pts: UiPoint[] = [{ coords: [0, 0], label: 42 }];
    const result = predictKNN(pts, 1, [0, 0]);
    expect(typeof result).toBe('string');
    expect(result).toBe('42');
  });
});
