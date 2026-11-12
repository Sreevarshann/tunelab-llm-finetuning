import { expect, test } from "@playwright/test";

/**
 * Full journey with REAL jobs: a tiny synthetic dataset, a 10-step LoRA run on the pinned base
 * model, automatic base vs LoRA evaluations, comparison, examples and the playground.
 */
test.describe.configure({ mode: "serial" });
const LONG = { timeout: 12 * 60_000 };
let runUrl = "";

test("upload validation reports line-level errors", async ({ page }) => {
  await page.goto("/datasets");
  await page.getByLabel("Dataset name").fill("broken");
  await page.getByLabel("JSONL file").setInputFiles({ name: "bad.jsonl", mimeType: "application/json", buffer: Buffer.from('{"messages":[{"role":"user","content":"hi"}]}\n{oops\n') });
  await page.getByRole("button", { name: "Upload" }).click();
  await expect(page.getByText("line 2: invalid JSON")).toBeVisible();
});

test("creates a small synthetic dataset through the API and inspects it", async ({ page, request }) => {
  const r = await request.post("/api/datasets/synthetic", { data: { name: "E2E triage", train: 40, valid: 8, test: 8, test_heldout: 8 } });
  expect(r.status()).toBe(201);
  await page.goto("/datasets");
  await page.getByRole("link", { name: /E2E triage/ }).click();
  await expect(page.getByText("synthetic", { exact: true }).first()).toBeVisible();
  await expect(page.getByRole("cell", { name: "Test · held-out templates" })).toBeVisible();
  await page.getByRole("tab", { name: "Test · held-out templates" }).click();
  await expect(page.getByText("Assistant (target)").first()).toBeVisible();
});

test("trains a real LoRA adapter with live loss and verification", async ({ page }) => {
  await page.goto("/runs/new");
  await page.getByLabel("Run name").fill("e2e tiny run");
  await page.getByRole("button", { name: /Quick/ }).click();
  await page.getByLabel("Steps").fill("10");
  await page.getByRole("button", { name: "Start training" }).click();
  await expect(page).toHaveURL(/\/runs\/run_\w+$/);
  runUrl = new URL(page.url()).pathname;
  await expect(page.getByRole("img", { name: "Training and validation loss by step" })).toBeVisible(LONG);
  await expect(page.getByText("Training verification")).toBeVisible(LONG);
  const norm = page.locator("dd", { hasText: "starts at 0" });
  await expect(norm).toContainText("0.000 →");
  const after = Number((await norm.textContent())!.match(/→\s*([\d.]+)/)![1]);
  expect(after).toBeGreaterThan(0); // weights were actually updated
  await expect(page.locator("dd", { hasText: /^[0-9a-f]{64}$/ })).toBeVisible();
});

test("evaluates base and LoRA on both test splits and pairs examples", async ({ page }) => {
  await page.goto(runUrl);
  await expect(page.getByText("Base model vs LoRA adapter")).toBeVisible();
  await expect(page.getByRole("img", { name: /Field accuracy, base vs LoRA, Test · held-out/ })).toBeVisible(LONG);
  await expect(page.getByRole("img", { name: "Field accuracy, base vs LoRA, Test", exact: true })).toBeVisible(LONG);
  await page.getByLabel("Example filter").selectOption("all");
  await expect(page.getByText("Reference").first()).toBeVisible();
  await expect(page.getByText("#0", { exact: true })).toBeVisible();
});

test("overview and compare show the run against the base model", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("link", { name: "e2e tiny run" })).toBeVisible();
  await expect(page.getByText(/vs base/).first()).toBeVisible();
  await page.goto("/compare");
  await expect(page.getByRole("img", { name: "Headline metric by run and split" })).toBeVisible();
});

test("playground runs base and base+LoRA on the same messages", async ({ page }) => {
  await page.goto("/playground");
  await page.getByRole("button", { name: "Run both" }).click();
  await expect(page.getByText("Base + LoRA")).toBeVisible(LONG);
  await expect(page.getByText("Base model", { exact: true })).toBeVisible();
});

test("queued runs can be cancelled", async ({ page, request }) => {
  const ds = (await (await request.get("/api/datasets")).json())[0];
  const a = await (await request.post("/api/runs", { data: { dataset_id: ds.id, name: "to cancel", config: { iters: 5 } } })).json();
  const b = await (await request.post("/api/runs", { data: { dataset_id: ds.id, name: "queued behind", config: { iters: 5 } } })).json();
  await page.goto(`/runs/${b.id}`);
  await page.getByRole("button", { name: "Cancel" }).click();
  await expect(page.getByText("Cancelled", { exact: true })).toBeVisible();
  await request.post(`/api/runs/${a.id}/cancel`);
});
