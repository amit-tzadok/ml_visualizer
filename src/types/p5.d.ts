/* Consolidated p5 types for the project.
   This file augments the 'p5' module with a permissive P5Instance interface
   tailored for demos and also exports the P5Instance type for local imports.
*/
import "p5";

export interface P5Instance {
  // Basic drawing / canvas
  createCanvas: (...args: Array<number | string>) => void;
  remove: () => void;
  width: number;
  height: number;
  background: (...args: Array<number | string>) => void;
  color: (...args: Array<number | string>) => unknown;
  lerpColor?: (c1: unknown, c2: unknown, t: number) => unknown;
  noStroke: () => void;
  fill: (...args: Array<number | string>) => void;
  circle: (...args: Array<number | string>) => void;
  rect: (...args: Array<number | string>) => void;
  map: (n: number, a: number, b: number, c: number, d: number) => number;
  random: (...args: Array<number>) => number;
  mouseX: number;
  mouseY: number;
  mouseButton?: number | string;
  LEFT?: number | string;

  // Lifecycle hooks typically used in sketches
  setup?: () => void;
  draw?: () => void;
  mousePressed?: () => void;
  windowResized?: () => void;

  // Offscreen graphics and helpers
  createGraphics?: (...args: Array<number>) => unknown;
  image?: (...args: Array<number | unknown>) => unknown;
  frameCount?: number;
  frameRate?: (n: number) => void;
  pixelDensity?: (n: number) => void;
  resizeCanvas?: (w: number, h: number) => void;
  touches?: Array<unknown>;

  // Our injected helpers
  updateDataset?: (dataset?: Array<{ x: number; y: number; label?: string }>) => void;
  trainMLP?: (opts?: { epochs?: number }) => void;
  resetDemo?: () => void;
  // demo-specific helpers
  speedScale?: number;
  maximizeMargin?: () => void;
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
  _mlpPointerHandler?: EventListener;
  _mlpContextHandler?: EventListener;
}

declare module "p5" {
  interface Instance extends P5Instance {}
}

export type { P5Instance as default };
