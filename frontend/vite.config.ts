import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";
import { VitePWA } from "vite-plugin-pwa";

export default defineConfig({
  plugins: [
    react(),
    // injectManifest strategy: vite-plugin-pwa builds + precaches the assets,
    // but our own src/sw.ts owns the service worker so we can implement
    // push + notificationclick handlers (which generateSW doesn't expose).
    VitePWA({
      strategies: "injectManifest",
      srcDir: "src",
      filename: "sw.ts",
      registerType: "autoUpdate",
      includeAssets: ["icons/*.png"],
      injectManifest: {
        globPatterns: ["**/*.{js,css,html,ico,png,svg}"],
      },
      manifest: {
        name: "BirdWatcher",
        short_name: "Birds",
        description: "AI bird-feeder camera + species ID",
        theme_color: "#2f5d50",
        background_color: "#f7f4ed",
        display: "standalone",
        start_url: "/",
        icons: [
          { src: "/icons/icon-192.png", sizes: "192x192", type: "image/png" },
          { src: "/icons/icon-512.png", sizes: "512x512", type: "image/png" },
          { src: "/icons/icon-512.png", sizes: "512x512", type: "image/png", purpose: "maskable" },
        ],
      },
    }),
  ],
  server: {
    // Port assigned in ~/Documents/Projects/PORTS.md; strictPort fails loudly
    // on collision instead of drifting. Backend stays on :8000 (Docker).
    port: Number(process.env.PORT) || 5270,
    strictPort: true,
    proxy: {
      "/api": "http://localhost:8000",
      "/media": "http://localhost:8000",
    },
  },
});
