export type Point = { x: number; y: number; label: string | number };
export type Dataset = Point[];

export type Predicted = {
  label: string | number | null;
  neighbors: number[];
  coords?: [number, number];
};

// minimal p5 instance type shape used by demos.
// avoid importing the full 'p5' types (they can be incompatible during incremental migration)
export type P5Graphics = {
  rect: (x: number, y: number, w: number, h: number) => void;
  // p5 Graphics buffers expose ellipse just like main instance; mark as present.
  ellipse: (x: number, y: number, w: number, h: number) => void;
  fill: (c: string | number | number[]) => void;
  noStroke: () => void;
  clear: () => void;
  getContext?: (type: string) => CanvasRenderingContext2D | WebGLRenderingContext | null;
  width?: number;
  height?: number;
};

export type P5Instance = {
  remove?: () => void;
  resetDemo?: () => void;
  createCanvas?: (w: number, h: number) => void;
  createGraphics?: (w: number, h: number) => P5Graphics;
  image?: (g: P5Graphics, x: number, y: number) => void;
  frameCount?: number;
  speedScale?: number;
  [key: string]: unknown;
};

// Lightweight adapter interface used by demos to interact with Perceptron-like models
export interface PerceptronAdapter {
  raw?: unknown;
  predict: (sample: number[]) => number;
  predictRaw: (sample: number[]) => number;
  trainSample: (sample: number[], label: number) => void;
  // optional faster training helpers
  trainSampleScore01?: (sample: number[], label01: 0 | 1, score: number) => void;
  trainSampleRaw01?: (sample: number[], label01: 0 | 1) => void;
  // metrics (optional)
  misclassificationRate?: (X: number[][], y: Array<number | string>) => number;
  hingeLoss?: (X: number[][], y: Array<number | string>) => number;
  // fit helpers
  fit?: (X: number[][], y: Array<number | string>) => void;
  fitHingeSGD?: (X: number[][], y: Array<number | string>, options?: { epochs?: number; lr?: number; lambda?: number; shuffle?: boolean }) => void;
  // parameters
  weights?: number[];
  bias?: number;
}

declare global {
  interface Window {
    mlvStatus?: {
      classifier?: string | null;
      equation?: string | null;
      weights?: number[] | undefined;
      bias?: number | undefined;
      updatedAt?: number;
    };
  }
}
