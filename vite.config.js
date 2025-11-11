import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
// We externalize `p5` only for production builds so the library is loaded at runtime
// from a CDN (see `src/utils/loadP5.ts`). During development we keep `p5` bundled
// so the dev server continues to work without relying on CDN availability.
export default defineConfig(({ mode }) => ({
  plugins: [react()],
  build: {
    rollupOptions: {
      // Externalize p5 in production to avoid bundling the large library into the app.
      external: mode === 'production' ? ['p5'] : [],
      output: {
        manualChunks: {
          // Split large libs so they can be cached separately and not block first paint
          react: ['react', 'react-dom'],
          katex: ['katex']
        }
      }
    }
  }
  // Uncomment and modify to use a specific port:
  // server: {
  //   port: 3000,
  //   strictPort: true,
  // },
}))
