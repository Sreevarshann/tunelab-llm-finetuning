import { defineConfig, devices } from "@playwright/test";

// E2E: isolated API (fresh data dir, real worker + MLX on the pinned base model) + production web build.
// Screenshots: SCREENSHOT_BASE_URL=http://localhost:3104 npx playwright test (the running stack).
const E2E = !process.env.SCREENSHOT_BASE_URL;
const API = "http://127.0.0.1:8194";

export default defineConfig({
  testDir: "./e2e",
  testMatch: E2E ? /.*\.e2e\.ts/ : /screenshots\.spec\.ts/,
  timeout: 15 * 60_000, // real training and generation; CPU-only CI is much slower than Metal
  expect: { timeout: 20_000 },
  workers: 1,
  retries: 0,
  reporter: [["list"], ["html", { open: "never" }]],
  use: { baseURL: process.env.SCREENSHOT_BASE_URL ?? "http://127.0.0.1:3194", trace: "retain-on-failure", colorScheme: "light" },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"], viewport: { width: 1440, height: 900 } } }],
  webServer: E2E
    ? [
        { command: "../backend/scripts/e2e-api.sh", url: `${API}/api/health`, timeout: 180_000, reuseExistingServer: false },
        { command: `NEXT_DIST_DIR=.next-e2e TUNELAB_API_ORIGIN=${API} npm run build && NEXT_DIST_DIR=.next-e2e TUNELAB_API_ORIGIN=${API} npx next start --port 3194 --hostname 127.0.0.1`, url: "http://127.0.0.1:3194", timeout: 300_000, reuseExistingServer: false },
      ]
    : undefined,
});
