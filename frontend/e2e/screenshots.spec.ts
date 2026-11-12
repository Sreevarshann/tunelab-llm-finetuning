/** Real screenshots of the running stack with the runs in eval/results.json. Not part of CI:
 *   SCREENSHOT_BASE_URL=http://localhost:3104 npx playwright test */
import { expect, type Page, test } from "@playwright/test";
import path from "node:path";

const out = (n: string) => path.join(__dirname, "..", "..", "docs", "screenshots", `${n}.png`);
const RUN = process.env.SHOT_RUN ?? "run_afb78b8ed7ee";
const settle = (p: Page, ms = 1500) => p.waitForTimeout(ms);

test("capture", async ({ page, browser }) => {
  test.setTimeout(300_000);
  await page.goto("/");
  await expect(page.getByRole("link", { name: /r8 · lr1e-4 · 300 steps/ })).toBeVisible();
  await settle(page);
  await page.screenshot({ path: out("01-runs") });

  await page.goto(`/runs/${RUN}`);
  await expect(page.getByRole("img", { name: /Field accuracy, base vs LoRA, Test · held-out/ })).toBeVisible();
  await settle(page, 2500);
  await page.screenshot({ path: out("02-run-loss-verification") });
  await page.getByText("Base model vs LoRA adapter").scrollIntoViewIfNeeded();
  await page.mouse.wheel(0, 380);
  await settle(page, 1500);
  await page.screenshot({ path: out("03-base-vs-lora") });
  await page.getByRole("button", { name: "Test · held-out templates" }).last().click();
  await page.getByLabel("Example filter").selectOption("tuned_wrong");
  await page.getByText("Examples: base vs LoRA").evaluate((el) => el.closest("div.rounded-xl")?.scrollIntoView({ block: "start" }));
  await settle(page, 1200);
  await page.screenshot({ path: out("04-examples-heldout-errors") });

  await page.goto("/compare");
  await page.setViewportSize({ width: 1440, height: 1100 });
  await expect(page.getByRole("img", { name: "Headline metric by run and split" })).toBeVisible();
  await settle(page, 2500);
  await page.screenshot({ path: out("05-compare") });
  await page.setViewportSize({ width: 1440, height: 900 });

  await page.goto("/datasets");
  await page.getByRole("link", { name: /Support ticket triage/ }).first().click();
  await expect(page.getByText("Assistant (target)").first()).toBeVisible();
  await settle(page);
  await page.screenshot({ path: out("06-dataset") });

  await page.goto("/runs/new");
  await settle(page);
  await page.screenshot({ path: out("07-new-run") });

  await page.goto(`/playground?run=${RUN}`);
  await page.getByRole("button", { name: "Run both" }).click();
  await expect(page.getByText("Base + LoRA")).toBeVisible({ timeout: 120_000 });
  await settle(page);
  await page.screenshot({ path: out("08-playground") });

  const dark = await browser.newContext({ viewport: { width: 1440, height: 900 }, colorScheme: "dark" });
  const d = await dark.newPage();
  await d.goto(`/runs/${RUN}`);
  await expect(d.getByRole("img", { name: /Field accuracy, base vs LoRA, Test · held-out/ })).toBeVisible();
  await settle(d, 2500);
  await d.screenshot({ path: out("09-run-dark") });
  await dark.close();

  const phone = await browser.newContext({ viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true });
  const m = await phone.newPage();
  await m.goto(`/runs/${RUN}`);
  await expect(m.getByText("Training verification")).toBeVisible();
  await settle(m, 2000);
  await m.screenshot({ path: out("10-mobile-run") });
  await phone.close();
});
