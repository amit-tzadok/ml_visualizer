import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    include: ['src/**/*.test.ts'],
    environment: 'node',
    globals: true
  },
  optimizeDeps: {
    // Ensure the SSR optimizer includes the TS file
    include: ['src/utils/mlp.ts']
  }
});
