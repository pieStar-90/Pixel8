type IssueRow = {
  issue_title?: string;
  issue_priority?: string;
  topic_group?: string;
  growth_ratio?: number;
  dominant_channel?: string;
};

function badge(p: string) {
  const s = (p || "").toLowerCase();
  if (s === "high") return "bg-red-500/15 text-red-800/60 border-red-800/50";
  if (s === "medium") return "bg-amber-500/15 text-amber-800/60 border-amber-800/50";
  return "bg-zinc-800/60 text-zinc-200 border-zinc-700/50";
}

import { Montserrat } from "next/font/google";

const mont = Montserrat({
  subsets: ["latin"],
  weight: ["700"], // pick what you use
});

export default function IssueTable({ rows }: { rows: IssueRow[] }) {
  return (
    <div className="overflow-hidden rounded-2xl border border-zinc-800">
      <table className="w-full text-left text-sm">
        <thead className="bg-[#7F7F7F] text-xs text-[#F0E0C8]">
          <tr>
            <th className={`${mont.className} text-sm px-4 py-3`}>Priority</th>
            <th className={`${mont.className} text-sm px-4 py-3`}>Issue</th>
            <th className={`${mont.className} text-sm px-4 py-3`}>Most common channel</th>
            <th className={`${mont.className} text-sm px-4 py-3`}>Growth</th>
          </tr>
        </thead>

        <tbody className="divide-y divide-zinc-800">
          {rows.map((r, idx) => (
            <tr key={idx} className="bg-[#eae9e7]">
              <td className="px-4 py-3">
                <span
                  className={`inline-flex items-center rounded-full border px-2 py-1 text-xs ${badge(
                    r.issue_priority ?? "high"
                  )}`}
                >
                  {(r.issue_priority ?? "high").toLowerCase()}
                </span>
              </td>

              <td className="px-4 py-3 font-medium text-zinc-600">
                {r.issue_title ?? "Untitled issue"}
              </td>

              <td className="px-4 py-3 text-zinc-600">{r.dominant_channel ?? "-"}</td>

              <td className="px-4 py-3 text-zinc-600">
                {typeof r.growth_ratio === "number" ? `${r.growth_ratio.toFixed(2)}×` : "-"}
              </td>
            </tr>
          ))}

          {!rows.length ? (
            <tr>
              <td className="px-4 py-6 text-zinc-400" colSpan={4}>
                No high priority issues right now.
              </td>
            </tr>
          ) : null}
        </tbody>
      </table>
    </div>
  );
}
