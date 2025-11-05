// Local p5 shim to override the package-provided types during incremental migration.
// This file is intentionally minimal and declares the module as `any` so the
// rest of the codebase can migrate without being blocked by upstream .d.ts
// parse issues.
declare module 'p5' {
  interface P5Constructor {
    new (sketch?: (p: import('../p5').P5Instance) => void, node?: Element | null): import('../p5').P5Instance;
  }
  const p5: P5Constructor;
  export default p5;
}

// Also provide a named export for CommonJS/ES interop imports like `import * as p5 from 'p5'`
declare module 'p5/lib/addons/p5.dom' {
  const whatever: unknown;
  export default whatever;
}
