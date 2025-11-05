export interface P5Instance {
  // Common p5 instance properties/methods used by our demos (kept permissive)
  createCanvas: (...args: Array<number | string>) => void;
  remove: () => void;
  map: (n: number, a: number, b: number, c: number, d: number) => number;
  width: number;
  height: number;
  // p5 lifecycle hooks we assign in sketches
  setup?: () => void;
  draw?: () => void;
  mousePressed?: () => void;
  mouseX: number;
  mouseY: number;
  mouseButton?: number | string;
  LEFT?: number | string;
  noStroke: () => void;
  fill: (...args: Array<number | string>) => void;
  circle: (...args: Array<number | string>) => void;
  rect: (...args: Array<number | string>) => void;
  background: (...args: Array<number | string>) => void;
  color: (...args: Array<number | string>) => unknown;
  lerpColor?: (c1: unknown, c2: unknown, t: number) => unknown;
  random: (...args: Array<number>) => number;
  createGraphics?: (...args: Array<number>) => unknown;
  frameRate?: (n: number) => void;
  pixelDensity?: (n: number) => void;
  resizeCanvas?: (w: number, h: number) => void;
  touches?: Array<unknown>;

  // our injected helpers
  updateDataset?: (dataset?: Array<{ x: number; y: number; label?: string }>) => void;
  trainMLP?: (opts?: { epochs?: number }) => void;
  resetDemo?: () => void;
  // internal/demo helpers used by sketches
  _mlpControls?: {
    optimizer?: string;
    activation?: string;
    lr?: number;
    batchSize?: number;
    epochs?: number;
    hiddenUnits?: number;
    touchClass?: "A" | "B";
    setGridStep?: (v: number) => void;
  };
  _lastLoss?: number;
  // allow attaching arbitrary handlers (pointer/context) for cleanup
  _mlpPointerHandler?: EventListener;
  _mlpContextHandler?: EventListener;
}
