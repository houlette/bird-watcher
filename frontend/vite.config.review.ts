// Throwaway config for in-browser review: proxies API/media to production
// so the local dev server renders real data. Not used by builds or deploys.
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

const PROD = "https://birdwatcher.ryanhoulette.com";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5273, // BirdWatcher block per ~/Documents/Projects/PORTS.md
    strictPort: true,
    proxy: {
      "/api": { target: PROD, changeOrigin: true, secure: true },
      "/media": { target: PROD, changeOrigin: true, secure: true },
    },
  },
});
