import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import devServer, { defaultOptions } from "@hono/vite-dev-server";

const isolationHeaders = {
  "Cross-Origin-Embedder-Policy": "require-corp",
  "Cross-Origin-Opener-Policy": "same-origin",
  "Permissions-Policy":
    "camera=(self), microphone=(), accelerometer=(self), gyroscope=(self), xr-spatial-tracking=(self)",
};

export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
    devServer({
      entry: "src/server/server.ts",
      exclude: [/.*\.(png|svg|jpeg|json|wasm)$/, ...defaultOptions.exclude],
    }),
  ],
  server: {
    allowedHosts: true,
    headers: isolationHeaders,
    host: true,
    port: 5555,
    strictPort: true,
  },
  preview: {
    headers: isolationHeaders,
    host: true,
    port: 5555,
    strictPort: true,
  },
});
