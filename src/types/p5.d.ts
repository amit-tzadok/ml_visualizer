/// <reference lib="dom" />
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
  stroke: (...args: Array<number | string>) => void;
  noFill: () => void;
  push: () => void;
  pop: () => void;
  translate: (...args: Array<number>) => void;
  rotate: (v: number) => void;
  line: (...args: Array<number>) => void;
  ellipse: (...args: Array<number>) => void;
  text: (...args: Array<number | string>) => void;
  textSize?: (n: number) => void;
  textAlign?: (...args: Array<unknown>) => void;
  CENTER?: number | string;
  strokeWeight?: (n: number) => void;
  red?: (c: unknown) => number;
  green?: (c: unknown) => number;
  blue?: (c: unknown) => number;
  clear?: () => void;
  key?: string;
  keyPressed?: () => void;
  noLoop?: () => void;
  loop?: () => void;
  // runtime helpers sketches may attach
  recomputeForK?: (k?: number) => void;
  requestRedraw?: () => void;
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
  // event handler types use DOM Event signature to avoid lint/no-undef issues
  _mlpPointerHandler?: (e: Event) => void;
  _mlpContextHandler?: (e: Event) => void;
}

declare module "p5" {
  // Replace an empty interface extension with a type alias to satisfy
  // `@typescript-eslint/no-empty-object-type` while preserving the augmentation.
  type Instance = P5Instance;
}

export type { P5Instance as default };
