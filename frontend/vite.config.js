import { defineConfig } from 'vite'
import { svelte } from '@sveltejs/vite-plugin-svelte'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [svelte()],
  base: '/static/',
  build: {
    outDir: '../static'
  },
  server: {
    port: '8080',
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:5000/'
      }
    }

  }
  
})
