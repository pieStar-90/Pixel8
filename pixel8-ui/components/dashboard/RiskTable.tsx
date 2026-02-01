// components/dashboard/RiskTable.tsx
export default function RiskTable({
  rows,
  onSelectRow,
}: {
  rows: Array<any>;
  onSelectRow: (idx: number) => void;
}) {
  const badge = (p: string) => {
    const s = (p || "").toLowerCase();
    if (s === "high") return "bg-red-500/15 text-red-200 border-red-800/50";
    if (s === "medium") return "bg-amber-500/15 text-amber-200 border-amber-800/50";
    return "bg-zinc-800/60 text-zinc-200 border-zinc-700/50";
  };

  return (
    <div className="overflow-hidden rounded-2xl border border-zinc-800">
      <table className="w-full text-left text-sm">
        <thead className="bg-zinc-950/40 text-xs text-zinc-300">
          <tr>
            <th className="px-4 py-3">Risk</th>
            <th className="px-4 py-3">Priority</th>
            <th className="px-4 py-3">Topic</th>
            <th className="px-4 py-3">Channel</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-zinc-800">
          {rows.map((r: any, idx: number) => (
            <tr
              key={idx}
              className="cursor-pointer bg-zinc-900/10 hover:bg-zinc-900/40"
              onClick={() => onSelectRow(idx)}
            >
              <td className="px-4 py-3 font-semibold">{r.risk_score}</td>
              <td className="px-4 py-3">
                <span className={`inline-flex items-center rounded-full border px-2 py-1 text-xs ${badge(r.priority)}`}>
                  {r.priority}
                </span>
                {r.review_required ? (
                  <span className="ml-2 inline-flex items-center rounded-full border border-orange-800/40 bg-orange-500/10 px-2 py-1 text-xs text-orange-200">
                    Review
                  </span>
                ) : null}
              </td>
              <td className="px-4 py-3">{r.topic_group}</td>
              <td className="px-4 py-3">
                {r.channel}
                {typeof r.likes === "number" ? <span className="text-zinc-400"> • {Math.round(r.likes)} likes</span> : null}
              </td>
              <td className="px-4 py-3">
                {r.sentiment ?? "neutral"}
                {typeof r.sentiment_confidence === "number" ? (
                  <span className="text-zinc-400"> • {(r.sentiment_confidence * 100).toFixed(0)}%</span>
                ) : null}
              </td>
              <td className="px-4 py-3 text-zinc-200">
                <div className="line-clamp-2 max-w-[680px]">{r.text}</div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
