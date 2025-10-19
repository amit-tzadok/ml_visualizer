// Thin re-export so that bundlers and Vitest resolve the TypeScript source.
// Re-export the TypeScript implementation directly to ensure bundlers
// and the test runner resolve the TS source instead of this JS file.
export { default } from './mlp.ts';
