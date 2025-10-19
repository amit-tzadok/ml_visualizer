import "p5";

declare module 'p5' {
  interface Instance {
    // dataset: array of points with x,y,label
    updateDataset?: (dataset?: Array<{ x: number; y: number; label?: string }>) => void;
    trainMLP?: (opts?: { epochs?: number }) => void;
    resetDemo?: () => void;
  }
}

// allow importing p5 as default
declare module 'p5' {
  const p5: any;
  export default p5;
}
