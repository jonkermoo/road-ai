import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react(),
    tailwindcss()
  ],
  server: {
    port: 5173,
    proxy: {
      "/video-feed": "http://localhost:8000",
      "/session": "http://localhost:8000",
      "/update-location": "http://localhost:8000",
      "/current-location": "http://localhost:8000",
      "/health": "http://localhost:8000",
      "/events": "http://localhost:8000",
      "/last-dets": "http://localhost:8000"
    }
  }
})
