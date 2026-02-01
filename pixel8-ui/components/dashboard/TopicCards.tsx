// components/dashboard/TopicCards.tsx
import { Montserrat } from "next/font/google";

const mont = Montserrat({
  subsets: ["latin"],
  weight: ["700"], // pick what you use
});

export default function TopicCards({
  topics,
  topicCounts,
}: {
  topics: Array<any>;
  topicCounts: Record<string, number>;
}) {
  return (
    <div className="rounded-2xl bg-white p-4 ring-1 ring-zinc-200 shadow-[0_0_30px_rgba(24,24,27,0.08)] transition-all duration-200 hover:ring-zinc-300 hover:shadow-[0_0_40px_rgba(24,24,27,0.14)]">
      <div className="flex items-end justify-between gap-3">
        <div>
          <h2 className={`${mont.className} text-lg font-semibold`}>Category Summary</h2>
        </div>
        <div className="text-xs text-zinc-400">
          Total topics: <span className="text-zinc-200">{Object.keys(topicCounts).length}</span>
        </div>
      </div>

      <div className="mt-4 grid grid-cols-1 gap-4 lg:grid-cols-3">
        {topics.map((t: any) => (
          <div key={t.topic_group} className="rounded-2xl  bg-[#eae9e7] p-4">
            <div className="flex items-start justify-between gap-3">
              <div>
                <div className="text-sm font-semibold">{t.topic_group}</div>
                <div className="mt-1 text-xs text-zinc-400">
                  Mentions: <span className="text-zinc-200">{t.mentions}</span>
                </div>
              </div>
              <div className="rounded-full bg-orange-500/15 px-3 py-1 text-xs text-orange-400">
                {t.growth_ratio.toFixed(2)}×
              </div>
            </div>

            <div className="mt-3 grid grid-cols-2 gap-3">
              <div className="rounded-xl bg-[#ed1047]/20 p-3">
                <div className="text-xs text-zinc-400">High</div>
                <div className="mt-1 text-lg font-semibold">{t.high_priority}</div>
              </div>
              <div className="rounded-xl bg-[#ffe178]/50 p-3">
                <div className="text-xs text-zinc-400">Medium</div>
                <div className="mt-1 text-lg font-semibold">{t.medium_priority}</div>
              </div>
            </div>

            <div className="mt-4">
              <div className="text-xs font-medium text-zinc-300">Examples</div>
              <ul className="mt-2 space-y-2 text-sm text-zinc-200">
                {(t.examples ?? []).slice(0, 2).map((ex: string, i: number) => (
                  <li key={i} className="rounded-xl bg-zinc-950/40 p-3">
                    {ex}
                  </li>
                ))}
              </ul>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
