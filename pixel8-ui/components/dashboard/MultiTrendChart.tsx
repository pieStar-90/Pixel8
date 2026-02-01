"use client";

import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid,
} from "recharts";
import { Montserrat } from "next/font/google";

const mont = Montserrat({
  subsets: ["latin"],
  weight: ["700"], // pick what you use
});

function fmtTick(iso: string) {
  const d = new Date(iso);
  const m = d.toLocaleString(undefined, { month: "short" });
  const day = d.getDate();
  const hh = String(d.getHours()).padStart(2, "0");
  return `${m} ${day} ${hh}:00`;
}

export default function MultiTrendChart({
  data,
}: {
  data: Array<{
    bucket_start_utc: string;
    service_outage: number;
    scam_rumour: number;
    negative_sentiment: number;
  }>;
}) {
  return (
    <div className="rounded-2xl bg-white">


      <div className="flex items-end justify-between gap-3">
        <div>
          <h2 className={`${mont.className} text-lg font-bold`}>Weekly Trends</h2>
        </div>
      </div>

      <div className="mt-4 h-72 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(161,161,170,0.5)" />

            <XAxis
              dataKey="bucket_start_utc"
              tickFormatter={fmtTick}
              minTickGap={22}
              stroke="rgba(63,63,70,0.9)"     // darker axis line
              tick={{ fontSize: 11, fill: "rgba(63,63,70,0.95)" }} // darker tick labels
            />

            <YAxis
              stroke="rgba(63,63,70,0.9)"     // darker axis line
              tick={{ fontSize: 11, fill: "rgba(63,63,70,0.95)" }} // darker tick labels
            />

            <Tooltip
              contentStyle={{
                background: "rgba(9,9,11,0.92)",
                border: "1px solid rgba(63,63,70,0.6)",
                borderRadius: 12,
                color: "white",
                fontSize: 12,
              }}
              labelFormatter={(v) => `Bucket: ${fmtTick(String(v))}`}
            />
            <Legend />

            <Line type="monotone" dataKey="service_outage" name="Service outage" stroke="#F97316" strokeWidth={2} dot={false} />
            <Line type="monotone" dataKey="scam_rumour" name="Scam rumour" stroke="#EF4444" strokeWidth={2} dot={false} />
            <Line type="monotone" dataKey="negative_sentiment" name="Negative sentiment" stroke="#A1A1AA" strokeWidth={2} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
