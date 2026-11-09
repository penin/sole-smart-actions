export function pct(x: number | null | undefined, digits = 1): string {
  if (x === null || x === undefined || Number.isNaN(x)) return "Not available";
  return `${(x * 100).toFixed(digits)}%`;
}

/** Threshold as a percentage: whole numbers for slider positions, 2 decimals for e.g. the 19.57% default. */
export function thresholdPct(t: number): string {
  const v = Math.round(t * 10000) / 100;
  return `${Number.isInteger(v) ? v.toFixed(0) : v.toFixed(2)}%`;
}

export function num(x: number | null | undefined, digits = 3): string {
  if (x === null || x === undefined || Number.isNaN(x)) return "Not available";
  return x.toFixed(digits);
}

/** TRANSFER_MONEY -> Transfer Money */
export function intentLabel(intent: string): string {
  return intent
    .toLowerCase()
    .split("_")
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(" ");
}
