/* Consolidated p5 types for the project.
   This file augments the 'p5' module with a permissive P5Instance interface
   tailored for demos and also exports the P5Instance type for local imports.
*/
import "p5";

export interface P5Instance {
  // Basic drawing / canvas
  createCanvas: (...args: any[]) => any;
  remove: () => void;
  width: number;
  height: number;
  background: (...args: any[]) => void;
  color: (...args: any[]) => any;
  lerpColor?: (c1: any, c2: any, t: number) => any;
  noStroke: () => void;
  fill: (...args: any[]) => void;
  circle: (...args: any[]) => void;
  rect: (...args: any[]) => void;
  map: (n: number, a: number, b: number, c: number, d: number) => number;
  random: (...args: any[]) => number;
  mouseX: number;
  mouseY: number;
  mouseButton?: any;
  LEFT?: any;

  // Lifecycle hooks typically used in sketches
  setup?: () => void;
  draw?: () => void;
  mousePressed?: () => void;

  // Offscreen graphics and helpers
  createGraphics?: (...args: any[]) => any;
  image?: (...args: any[]) => any;
  frameCount?: number;

  // Our injected helpers
  updateDataset?: (dataset?: Array<{ x: number; y: number; label?: string }>) => void;
  trainMLP?: (opts?: { epochs?: number }) => void;
  resetDemo?: () => void;
  // demo-specific helpers
  speedScale?: number;
  maximizeMargin?: () => void;
}

declare module "p5" {
  interface Instance extends P5Instance {}
}

export type { P5Instance as default };
