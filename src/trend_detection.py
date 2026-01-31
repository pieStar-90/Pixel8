import os

# Make sure the output folder exists
os.makedirs("data/processed", exist_ok=True)

import pandas as pd
from datetime import timedelta

df = pd.read_csv("data/synthesised_posts.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

TOPIC_KEYWORDS = {
    "service_outage": ["login", "crash", "down", "not working", "error", "failed"],
    "scam_rumor": ["scam", "fraud", "fake", "phishing", "otp", "call"],
    "brand_sentiment": ["support", "customer service", "helpdesk", "response"],
}

NEGATIVE_WORDS = ["not", "never", "bad", "worst", "angry", "delay", "failed", "issue"]
POSITIVE_WORDS = ["good", "great", "smooth", "fast", "helpful"]

def infer_topic(text):
    text = text.lower()
    for topic, keywords in TOPIC_KEYWORDS.items():
        for word in keywords:
            if word in text:
                return topic
    return "general_feedback"

df["topic"] = df["text"].apply(infer_topic)

def infer_sentiment(text):
    text = text.lower()
    neg = sum(word in text for word in NEGATIVE_WORDS)
    pos = sum(word in text for word in POSITIVE_WORDS)

    if neg > pos:
        return "negative"
    elif pos > neg:
        return "positive"
    else:
        return "neutral"

df["sentiment"] = df["text"].apply(infer_sentiment)

now = df["timestamp"].max()

last_24h = df[df["timestamp"] >= now - timedelta(hours=24)]
prev_24h = df[
    (df["timestamp"] >= now - timedelta(hours=48)) &
    (df["timestamp"] < now - timedelta(hours=24))
]

def detect_trends(topic):
    curr = last_24h[last_24h["topic"] == topic]
    prev = prev_24h[prev_24h["topic"] == topic]

    curr_vol = len(curr)
    prev_vol = len(prev)

    spike_ratio = curr_vol / prev_vol if prev_vol > 0 else curr_vol

    curr_neg = len(curr[curr["sentiment"] == "negative"])
    prev_neg = len(prev[prev["sentiment"] == "negative"])

    curr_neg_ratio = curr_neg / curr_vol if curr_vol > 0 else 0
    prev_neg_ratio = prev_neg / prev_vol if prev_vol > 0 else 0

    sentiment_shift = curr_neg_ratio - prev_neg_ratio

    return {
        "topic": topic,
        "current_volume": curr_vol,
        "previous_volume": prev_vol,
        "spike_ratio": round(spike_ratio, 2),
        "negative_sentiment_change": round(sentiment_shift, 2),
        "trend_flag": spike_ratio >= 2 or sentiment_shift > 0.2
    }

# Step 1: run detect_trends for all topics
topics = df["topic"].unique()
trend_results = [detect_trends(t) for t in topics]

# Optional: convert to a DataFrame to save as CSV
trend_df = pd.DataFrame(trend_results)

# Save the CSV into data/processed/
trend_df.to_csv("data/processed/grouped_posts.csv", index=False)

print("Step 1 complete: trend CSV saved to data/processed/grouped_posts.csv")
