import type { NextConfig } from "next";

const config: NextConfig = {
  output: "standalone",
  experimental: { proxyClientMaxBodySize: "34mb" },
  async rewrites() {
    const api = process.env.API_INTERNAL_URL ?? "http://127.0.0.1:8000";
    return [{ source: "/api/:path*", destination: `${api}/api/:path*` }];
  },
};

export default config;
