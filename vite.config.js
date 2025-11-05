import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  // Uncomment and modify to use a specific port:
  // server: {
  //   port: 3000,
  //   strictPort: true,
  // },
})
