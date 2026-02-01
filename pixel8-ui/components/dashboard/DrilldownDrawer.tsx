"use client";

export default function DrilldownDrawer({
  open,
  onClose,
  post,
}: {
  open: boolean;
  onClose: () => void;
  post: any | null;
}) {
  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50">
      {/* backdrop */}
      <div className="absolute inset-0 bg-black/60" onClick={onClose} />

      {/* panel */}
      <div className="absolute right-0 top-0 h-full w-full max-w-xl overflow-y-auto border-l border-zinc-800 bg-zinc-950 p-6">
        <div className="flex items-start justify-between gap-3">
          <div>
            <div className="text-xs text-zinc-400">Risk Detail</div>
            <div className="mt-1 text-lg font-semibold">{post?.topic_group ?? "—"}</div>
          </div>
          <button
            className="rounded-xl border border-zinc-800 bg-zinc-900/40 px-3 py-2 text-sm text-zinc-200 hover:bg-zinc-900/70"
            onClick={onClose}
          >
            Close
          </button>
        </div>

        <div className="mt-5 space-y-4">
          <div className="rounded-2xl border border-zinc-800 bg-zinc-900/30 p-4">
            <div className="text-xs text-zinc-400">Post</div>
            <div className="mt-2 text-sm leading-relaxed text-zinc-100">{post?.text ?? "—"}</div>
            <div className="mt-3 text-xs text-zinc-400">
              {post?.timestamp ?? "—"} • {post?.channel ?? "—"}
              {typeof post?.likes === "number" ? ` • ${Math.round(post.likes)} likes` : ""}
            </div>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <Metric label="Risk score" value={post?.risk_score} />
            <Metric label="Priority" value={post?.priority} />
            <Metric label="Confidence" value={fmtPct(post?.confidence)} />
            <Metric label="Uncertainty" value={fmtPct(post?.uncertainty)} />
          </div>

          <div className="rounded-2xl border border-zinc-800 bg-zinc-900/30 p-4">
            <div className="text-xs font-medium text-zinc-300">Why flagged</div>
            <div className="mt-2 text-sm text-zinc-100">
              {post?.risk_drivers ?? "—"}
            </div>
          </div>

          <div className="rounded-2xl border border-zinc-800 bg-zinc-900/30 p-4">
            <div className="text-xs font-medium text-zinc-300">Suggested actions (demo)</div>
            <div className="mt-3 flex flex-wrap gap-2">
              <ActionBtn label="Create incident ticket" />
              <ActionBtn label="Notify comms" />
              <ActionBtn label="Escalate to fraud team" />
              <ActionBtn label="Mark resolved" />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: any }) {
  return (
    <div className="rounded-2xl border border-zinc-800 bg-zinc-900/30 p-4">
      <div className="text-xs text-zinc-400">{label}</div>
      <div className="mt-1 text-lg font-semibold">{value ?? "—"}</div>
    </div>
  );
}

function ActionBtn({ label }: { label: string }) {
  return (
    <button className="rounded-xl border border-zinc-800 bg-zinc-900/40 px-3 py-2 text-sm text-zinc-200 hover:bg-zinc-900/70">
      {label}
    </button>
  );
}

function fmtPct(x?: number) {
  if (typeof x !== "number" || Number.isNaN(x)) return "—";
  return `${Math.round(x * 100)}%`;
}
