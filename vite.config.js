import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          // Split large libs so they can be cached separately and not block first paint
          react: ['react', 'react-dom'],
          p5: ['p5'],
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
})
