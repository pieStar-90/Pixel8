"use client";

export default function TrendInsightDrawer({
  open,
  onClose,
  loading,
  error,
  insight,
  onRegenerate,
}: {
  open: boolean;
  onClose: () => void;
  loading: boolean;
  error: string | null;
  insight: any | null;
  onRegenerate: () => void;
}) {
  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50">
      <div className="absolute inset-0 bg-black/70" onClick={onClose} />
      <div className="absolute right-0 top-0 h-full w-full max-w-xl border-l border-zinc-800 bg-zinc-950 p-6">
        <div className="flex items-start justify-between gap-3">
          <div>
            <div className="text-xs text-zinc-400">Weekly Trends Insight</div>
            <h3 className="mt-1 text-lg font-semibold">AI Briefing</h3>
            {insight?.generated_at_utc && (
              <div className="mt-1 text-xs text-zinc-400">
                Generated: <span className="text-zinc-200">{insight.generated_at_utc}</span>
              </div>
            )}
          </div>

          <div className="flex gap-2">
            <button
              onClick={onRegenerate}
              className="rounded-lg border border-zinc-800 bg-zinc-900/40 px-3 py-2 text-xs hover:bg-zinc-900/60"
            >
              Regenerate
            </button>
            <button
              onClick={onClose}
              className="rounded-lg border border-zinc-800 bg-zinc-900/40 px-3 py-2 text-xs hover:bg-zinc-900/60"
            >
              Close
            </button>
          </div>
        </div>

        <div className="mt-5 space-y-5">
          {loading && (
            <div className="rounded-xl border border-zinc-800 bg-zinc-900/30 p-4 text-sm text-zinc-300">
              Generating insight…
            </div>
          )}

          {error && (
            <div className="rounded-xl border border-red-900/50 bg-red-950/30 p-4 text-sm text-red-200">
              {error}
            </div>
          )}

          {!loading && insight && (
            <>
              <div className="rounded-2xl border border-zinc-800 bg-zinc-900/30 p-4">
                <div className="text-xs text-zinc-400">Headline</div>
                <div className="mt-1 text-base font-semibold">{insight.headline}</div>
                <div className="mt-3 text-sm text-zinc-200">{insight.summary}</div>
                {insight.confidence_note && (
                  <div className="mt-3 rounded-xl border border-zinc-800 bg-zinc-900/40 p-3 text-xs text-zinc-300">
                    <span className="font-medium text-zinc-200">Confidence:</span>{" "}
                    {insight.confidence_note}
                  </div>
                )}
              </div>

              <div className="grid gap-4">
                <Section title="Key points" items={insight.bullets} />
                <Section title="Potential causes" items={insight.potential_causes} />
                <Section title="Recommended actions" items={insight.recommended_actions} />
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

function Section({ title, items }: { title: string; items?: string[] }) {
  const arr = Array.isArray(items) ? items : [];
  return (
    <div className="rounded-2xl border border-zinc-800 bg-zinc-900/30 p-4">
      <div className="text-sm font-semibold">{title}</div>
      <ul className="mt-3 space-y-2 text-sm text-zinc-200">
        {arr.length ? (
          arr.map((t, i) => (
            <li key={i} className="flex gap-2">
              <span className="mt-1 h-1.5 w-1.5 flex-none rounded-full bg-orange-400" />
              <span>{t}</span>
            </li>
          ))
        ) : (
          <li className="text-zinc-400">No items</li>
        )}
      </ul>
    </div>
  );
}
