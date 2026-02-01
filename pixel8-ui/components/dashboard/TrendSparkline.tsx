// components/dashboard/TrendSparkline.tsx
"use client";

import { ResponsiveContainer, LineChart, Line, Tooltip } from "recharts";

export default function TrendSparkline({
  previous,
  recent,
}: {
  previous: number;
  recent: number;
}) {
  const data = [
    { name: "Prev", value: previous },
    { name: "Recent", value: recent },
  ];

  return (
    <div className="h-12 w-32">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 6, right: 6, left: 6, bottom: 0 }}>
          <Tooltip
            contentStyle={{
              background: "rgba(9,9,11,0.9)",
              border: "1px solid rgba(63,63,70,0.6)",
              borderRadius: 12,
              color: "white",
              fontSize: 12,
            }}
            labelStyle={{ color: "rgba(228,228,231,0.85)" }}
            formatter={(v: any) => [v, "Mentions"]}
          />
          <Line
            type="monotone"
            dataKey="value"
            strokeWidth={2}
            dot={{ r: 2 }}
            activeDot={{ r: 4 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
