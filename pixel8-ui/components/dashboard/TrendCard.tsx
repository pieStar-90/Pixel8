// components/dashboard/TrendCard.tsx
import { trendLabel } from "@/lib/dashboard";
import TrendSparkline from "./TrendSparkline";

export default function TrendCard({
  title,
  subtitle,
  value,
}: {
  title: string;
  subtitle?: string;
  value?: { recent: number; previous: number; growth_ratio: number };
}) {
  const v = value ?? { recent: 0, previous: 0, growth_ratio: 1.0 };
  const t = trendLabel(v.growth_ratio);

  return (
    <div className="rounded-2xl border border-zinc-800 bg-zinc-900/30 p-4">
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="text-sm font-medium">{title}</div>
          {subtitle && <div className="mt-1 text-xs text-zinc-400">{subtitle}</div>}
        </div>

        <div className="flex items-center gap-3">
          <TrendSparkline previous={v.previous} recent={v.recent} />

          <div className="rounded-full bg-zinc-800/60 px-3 py-1 text-xs text-zinc-200">
            {t.glyph} {t.label}
          </div>
        </div>
      </div>

      <div className="mt-4 grid grid-cols-3 gap-3">
        <div className="rounded-xl bg-zinc-950/40 p-3">
          <div className="text-xs text-zinc-400">Recent</div>
          <div className="mt-1 text-lg font-semibold">{v.recent}</div>
        </div>
        <div className="rounded-xl bg-zinc-950/40 p-3">
          <div className="text-xs text-zinc-400">Previous</div>
          <div className="mt-1 text-lg font-semibold">{v.previous}</div>
        </div>
        <div className="rounded-xl bg-zinc-950/40 p-3">
          <div className="text-xs text-zinc-400">Growth</div>
          <div className="mt-1 text-lg font-semibold">{v.growth_ratio.toFixed(2)}×</div>
        </div>
      </div>
    </div>
  );
}
