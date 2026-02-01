// lib/dashboard.ts
import type { DashboardSummary } from "@/types/dashboard";

export async function fetchDashboardSummary(): Promise<DashboardSummary> {
  const res = await fetch("/api/dashboard", { cache: "no-store" });
  if (!res.ok) throw new Error("Failed to load dashboard summary");
  return res.json();
}

export function formatPct(x?: number) {
  if (typeof x !== "number" || Number.isNaN(x)) return "—";
  return `${Math.round(x * 100)}%`;
}

export function trendLabel(growthRatio: number) {
  if (growthRatio >= 1.2) return { label: "Rising", glyph: "▲" };
  if (growthRatio <= 0.8) return { label: "Falling", glyph: "▼" };
  return { label: "Stable", glyph: "▬" };
}
