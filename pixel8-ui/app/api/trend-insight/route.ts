import { NextResponse } from "next/server";
import path from "path";
import { promises as fs } from "fs";
import { GoogleGenerativeAI } from "@google/generative-ai";

export const runtime = "nodejs";

type Point = {
  bucket_start_utc: string;
  service_outage: number;
  scam_rumour: number;
  negative_sentiment: number;
};

function pickKeyStats(series: Point[]) {
  if (!series?.length) {
    return {
      lastBucket: null,
      last24h: { service_outage: 0, scam_rumour: 0, negative_sentiment: 0 },
      prev24h: { service_outage: 0, scam_rumour: 0, negative_sentiment: 0 },
    };
  }

  const n = series.length;
  const last = series[n - 1];

  // 6-hour buckets: 4 buckets = 24h
  const last24 = series.slice(Math.max(0, n - 4));
  const prev24 = series.slice(Math.max(0, n - 8), Math.max(0, n - 4));

  const sum = (arr: Point[], k: keyof Point) =>
    arr.reduce((acc, p) => acc + Number(p[k] ?? 0), 0);

  return {
    lastBucket: last,
    last24h: {
      service_outage: sum(last24, "service_outage"),
      scam_rumour: sum(last24, "scam_rumour"),
      negative_sentiment: sum(last24, "negative_sentiment"),
    },
    prev24h: {
      service_outage: sum(prev24, "service_outage"),
      scam_rumour: sum(prev24, "scam_rumour"),
      negative_sentiment: sum(prev24, "negative_sentiment"),
    },
  };
}

export async function GET() {
  try {
    // Read backend summary JSON (single source of truth)
    const filePath = path.join(process.cwd(), "..", "data", "dashboard_summary.json");
    const raw = await fs.readFile(filePath, "utf8");
    const dashboard = JSON.parse(raw);

    const series: Point[] = Array.isArray(dashboard.timeseries_6h_7d)
      ? dashboard.timeseries_6h_7d
      : [];
    const topPosts = Array.isArray(dashboard.top_posts)
      ? dashboard.top_posts.slice(0, 12)
      : [];
    const stats = pickKeyStats(series);

    if (!process.env.GEMINI_API_KEY) {
      return NextResponse.json(
        { error: "Missing GEMINI_API_KEY in pixel8-ui/.env.local" },
        { status: 500 }
      );
    }

    const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
    const model = genAI.getGenerativeModel({
      // Pick a Gemini model available to your key (this one is commonly used)
      model: "gemini-3-flash-preview",
    });

    const payload = {
      context: {
        buckets: "6-hour buckets for last 7 days",
        metrics: ["service_outage", "scam_rumour", "negative_sentiment"],
        note:
          "Negative sentiment volume may be skewed; focus on changes/spikes and plausible causes. " +
          "Do not invent internal bank system facts; base reasoning only on provided metrics + example posts.",
      },
      key_stats: stats,
      timeseries_6h_7d: series,
      example_posts: topPosts.map((p: any) => ({
        timestamp: p.timestamp,
        channel: p.channel,
        topic_group: p.topic_group,
        sentiment: p.sentiment,
        risk_score: p.risk_score,
        text: p.text,
      })),
    };

    const prompt = `
You are a bank-grade risk intelligence analyst writing an executive trend insight for Mashreq Signals.

Return ONLY valid JSON with this exact schema:
{
  "headline": string,
  "summary": string,
  "bullets": string[],
  "potential_causes": string[],
  "recommended_actions": string[],
  "confidence_note": string
}

Rules:
- Use only the provided public-chatter metrics and example posts.
- Be concise, credible, and specific about directionality (rising/falling/spike).
- Do NOT claim internal outages/confirmations; say "may indicate" / "could be consistent with".
- 4–8 bullets max, 3–6 causes, 3–6 actions.

Data:
${JSON.stringify(payload, null, 2)}
`.trim();

    const result = await model.generateContent(prompt);
    const text = result.response.text().trim();

    // Gemini sometimes wraps JSON in ```json fences — strip them safely
    const cleaned = text
      .replace(/^```json\s*/i, "")
      .replace(/^```\s*/i, "")
      .replace(/```$/i, "")
      .trim();

    const parsed = JSON.parse(cleaned);

    return NextResponse.json(
      {
        generated_at_utc: new Date().toISOString(),
        window: "6h buckets, 7 days",
        ...parsed,
      },
      { status: 200 }
    );
  } catch (err: any) {
    return NextResponse.json(
      { error: "Failed to generate Gemini trend insight", details: err?.message ?? String(err) },
      { status: 500 }
    );
  }
}
