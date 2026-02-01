// types/dashboard.ts

export type Priority = "high" | "medium" | "low";
export type TopicGroup = "service_outage" | "scam_rumour" | "general_feedback";
export type Sentiment = "negative" | "neutral" | "positive";

export interface TrendValue {
  recent: number;
  previous: number;
  growth_ratio: number;
}

export interface DashboardSummary {
  generated_at_utc: string;
  windows: {
    recent_hours: number;
    previous_hours: number;
  };
  totals: {
    rows: number;
    priority_counts: Record<Priority, number> | Record<string, number>;
    topic_group_counts: Record<TopicGroup, number> | Record<string, number>;
    review_required_count: number;
    high_priority_issues: number;
    medium_priority_issues: number;
  };
  trends: {
    topic_group: Record<TopicGroup, TrendValue> | Record<string, TrendValue>;
    sentiment: Record<Sentiment, TrendValue> | Record<string, TrendValue>;
  };
  top_topics: Array<{
    topic_group: TopicGroup | string;
    mentions: number;
    recent_mentions: number;
    previous_mentions: number;
    growth_ratio: number;
    avg_risk: number;
    max_risk: number;
    avg_confidence: number;
    high_priority: number;
    medium_priority: number;
    examples: string[];
  }>;
  top_posts: Array<{
    timestamp: string;
    channel: string;
    likes?: number;
    text: string;

    topic_group: TopicGroup | string;
    topic_group_confidence?: number;

    sentiment?: Sentiment | string;
    sentiment_confidence?: number;

    certainty?: number;
    confidence?: number;
    uncertainty?: number;

    risk_score: number;
    priority: Priority | string;
    review_required?: boolean;

    risk_drivers?: string;
  }>;
    timeseries_6h_7d: Array<{
    bucket_start_utc: string;
    service_outage: number;
    scam_rumour: number;
    negative_sentiment: number;
  }>;

}
