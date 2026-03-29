import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    include: ['src/**/*.test.ts'],
    environment: 'node',
    globals: true,
    coverage: {
      provider: 'v8',
      include: ['src/utils/**/*.ts'],
      exclude: [
        'src/utils/**/*.test.ts',
        'src/utils/loadP5.ts',
        'src/utils/embeddings.ts',
        'src/utils/rag.ts',
      ],
      reporter: ['text', 'json-summary', 'lcov'],
      thresholds: {
        lines: 70,
        functions: 70,
        branches: 55,
        statements: 70,
      },
    },
  },
  optimizeDeps: {
    // Ensure the SSR optimizer includes the TS file
    include: ['src/utils/mlp.ts']
  }
});
