/** Mirrors backend scoring.parse_json_object: an object alone or in a ```json fence, nothing repaired. */
export function parseMaybeJson(text: string): Record<string, unknown> | null {
  const t = text.trim().replace(/^```(?:json)?\s*/m, "").replace(/\s*```$/m, "").trim();
  try {
    const v = JSON.parse(t);
    return v && typeof v === "object" && !Array.isArray(v) ? v : null;
  } catch {
    return null;
  }
}
