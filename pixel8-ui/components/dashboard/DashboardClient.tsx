"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import type { DashboardSummary } from "@/types/dashboard";
import { fetchDashboardSummary } from "@/lib/dashboard";

import KpiCard from "./kpiCard";
import TrendCard from "./TrendCard";
import TopicCards from "./TopicCards";
import MultiTrendChart from "./MultiTrendChart";
import TrendInsightDrawer from "./TrendInsightDrawer";
import IssueTable from "./IssueTable";
import { Montserrat } from "next/font/google";

const mont = Montserrat({
  subsets: ["latin"],
  weight: ["700"], // pick what you use
});

export default function DashboardClient() {
  const [data, setData] = useState<DashboardSummary | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Scroll target
  const priorityQueueRef = useRef<HTMLDivElement | null>(null);

  // Trend insight drawer state
  const [trendOpen, setTrendOpen] = useState(false);
  const [trendInsight, setTrendInsight] = useState<any>(null);
  const [trendLoading, setTrendLoading] = useState(false);
  const [trendError, setTrendError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;

    const load = async () => {
      try {
        const d = await fetchDashboardSummary();
        if (alive) setData(d);
      } catch (e: any) {
        if (alive) setError(e?.message ?? "Failed to load");
      }
    };

    load();
    const id = setInterval(load, 5000);

    return () => {
      alive = false;
      clearInterval(id);
    };
  }, []);

  // Top 10 high-priority issues, aggregate only
  const topHighIssues = useMemo(() => {
    const issues = (data as any)?.top_issues ?? [];
    return issues
      .filter((x: any) => String(x?.issue_priority ?? "").toLowerCase() === "high")
      .slice(0, 5);
  }, [data]);

  async function loadTrendInsight() {
    setTrendLoading(true);
    setTrendError(null);
    try {
      const r = await fetch("/api/trend-insight", { cache: "no-store" });
      if (!r.ok) throw new Error(await r.text());
      const j = await r.json();
      setTrendInsight(j);
    } catch (e: any) {
      setTrendError(e?.message ?? "Failed to generate insight");
    } finally {
      setTrendLoading(false);
    }
  }

  function openTrendDrawer() {
    setTrendOpen(true);
    if (!trendInsight && !trendLoading) loadTrendInsight();
  }

  function scrollToPriorityQueue() {
    priorityQueueRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  if (error) {
    return (
      <div className="rounded-2xl border border-red-900/50 bg-red-950/30 p-6">
        <div className="text-sm text-red-200">Error</div>
        <div className="mt-1 font-medium">{error}</div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="rounded-2xl border border-zinc-800 bg-zinc-900/30 p-6">
        <div className="text-sm text-zinc-300">Loading dashboard…</div>
      </div>
    );
  }

  const priority = data.totals.priority_counts ?? {};
  const topicCounts = data.totals.topic_group_counts ?? {};
  const reviewCount = data.totals.review_required_count ?? 0;

  const highIssues = (data.totals as any).high_priority_issues ?? 0;
  const mediumIssues = (data.totals as any).medium_priority_issues ?? 0;

  const ts = data.timeseries_6h_7d ?? [];
  const hasTimeseries = Array.isArray(ts) && ts.length > 0;

  return (
    <div className="space-y-6">
      {/* KPI row */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <KpiCard
          title="High priority issues"
          value={highIssues}
          tone="danger"
          onClick={scrollToPriorityQueue}
        />
        <KpiCard title="Medium priority issues" value={mediumIssues} tone="warning" />
        <KpiCard title="Low priority posts" value={priority["low"] ?? 0} tone="neutral" />
        <KpiCard title="Human review" value={reviewCount} tone="accent" />
      </div>

      {/* Weekly multi-line trends (6h buckets, 7 days) */}
      {hasTimeseries ? (
        <div
          role="button"
          tabIndex={0}
          onClick={openTrendDrawer}
          className="cursor-pointer rounded-2xl bg-white p-4 ring-1 ring-zinc-200 shadow-[0_0_30px_rgba(24,24,27,0.08)] transition-all duration-200 hover:ring-zinc-300 hover:shadow-[0_0_40px_rgba(24,24,27,0.14)]"
        >
          <MultiTrendChart data={data.timeseries_6h_7d} />
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
          <TrendCard
            title="Service outage trend"
            subtitle={`Last ${data.windows.recent_hours}h vs previous ${data.windows.previous_hours}h`}
            value={data.trends.topic_group["service_outage"]}
          />
          <TrendCard
            title="Scam rumour trend"
            subtitle={`Last ${data.windows.recent_hours}h vs previous ${data.windows.previous_hours}h`}
            value={data.trends.topic_group["scam_rumour"]}
          />
          <TrendCard
            title="Negative sentiment trend"
            subtitle={`Last ${data.windows.recent_hours}h vs previous ${data.windows.previous_hours}h`}
            value={data.trends.sentiment["negative"]}
          />
        </div>
      )}

      {/* Top topics */}
      <TopicCards topics={data.top_topics} topicCounts={topicCounts} />

      {/* Live Priority Queue (issues, aggregate only) */}
      <div
        ref={priorityQueueRef}
        className="rounded-2xl bg-white p-4 ring-1 ring-zinc-200 shadow-[0_0_30px_rgba(24,24,27,0.08)] transition-all duration-200 hover:ring-zinc-300 hover:shadow-[0_0_40px_rgba(24,24,27,0.14)]"
      >
        <div className="flex items-end justify-between gap-3">
          <div>
            <h2 className={`${mont.className} text-lg font-bold`}>Live Priority Queue</h2>
            <p className="text-sm text-zinc-600">
              Top 5 priority issues (aggregated by topic)
            </p>
          </div>
          <div className="text-xs text-zinc-500">
            Generated: <span className="text-zinc-900">{data.generated_at_utc}</span>
          </div>
        </div>

        <div className="mt-4">
          <IssueTable rows={topHighIssues} />
        </div>
      </div>

      <TrendInsightDrawer
        open={trendOpen}
        onClose={() => setTrendOpen(false)}
        loading={trendLoading}
        error={trendError}
        insight={trendInsight}
        onRegenerate={loadTrendInsight}
      />
    </div>
  );
}
