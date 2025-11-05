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
