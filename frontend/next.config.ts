import type { NextConfig } from "next";

const API_ORIGIN = process.env.TUNELAB_API_ORIGIN ?? "http://127.0.0.1:8114";

const nextConfig: NextConfig = {
  cacheComponents: true,
  agentRules: false,
  // separate build output lets the E2E production build run beside `next dev`
  distDir: process.env.NEXT_DIST_DIR ?? ".next",
  rewrites() {
    return [{ source: "/api/:path*", destination: `${API_ORIGIN}/api/:path*` }];
  },
  turbopack: { rules: { "*.css": { loaders: ["@tailwindcss/turbopack"], as: "*.css" } } },
};

export default nextConfig;
