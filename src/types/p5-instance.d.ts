export interface P5Instance {
  // Common p5 instance properties/methods used by our demos (kept permissive)
  createCanvas: (...args: any[]) => any;
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
  mouseButton?: any;
  LEFT?: any;
  noStroke: () => void;
  fill: (...args: any[]) => void;
  circle: (...args: any[]) => void;
  rect: (...args: any[]) => void;
  background: (...args: any[]) => void;
  color: (...args: any[]) => any;
  lerpColor?: (c1: any, c2: any, t: number) => any;
  random: (...args: any[]) => number;
  createGraphics?: (...args: any[]) => any;

  // our injected helpers
  updateDataset?: (dataset?: Array<{ x: number; y: number; label?: string }>) => void;
  trainMLP?: (opts?: { epochs?: number }) => void;
  resetDemo?: () => void;
}
