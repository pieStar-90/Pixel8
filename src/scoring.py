# src/scoring.py

import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.sentiment import SentimentIntensityAnalyzer
import os

# ---- Setup folders ----
os.makedirs("data/processed", exist_ok=True)

# ---- Load the trend detection CSV ----
df = pd.read_csv("data/processed/grouped_posts.csv")

# If your CSV has a different text column name, adjust here
if "text" not in df.columns:
    df["text"] = df["Post"]  # Example if your column is Post

# ---- Initialize sentiment analyzer ----
sia = SentimentIntensityAnalyzer()

# ---- Functions ----

def negativity_score(texts):
    """Returns fraction of negative posts"""
    if len(texts) == 0:
        return 0
    scores = [sia.polarity_scores(t)["compound"] for t in texts]
    negative_count = len([s for s in scores if s < -0.2])
    return negative_count / len(texts)

def wording_similarity(texts):
    """Returns average cosine similarity between posts"""
    if len(texts) < 2:
        return 1.0  # Single post = max similarity
    tfidf = TfidfVectorizer(stop_words="english")
    matrix = tfidf.fit_transform(texts)
    sim = cosine_similarity(matrix)
    return sim.mean()

def compute_signal(topic_df):
    """Compute a Signal Object from posts of a single topic"""
    texts = topic_df["text"].tolist()
    volume = len(texts)

    # Speed: compare current vs previous volume from trend detection columns
    curr_vol = topic_df["current_volume"].iloc[0] if "current_volume" in topic_df else volume
    prev_vol = topic_df["previous_volume"].iloc[0] if "previous_volume" in topic_df else max(volume - 5, 1)
    speed = curr_vol / max(prev_vol, 1)

    # Negativity score
    negativity = negativity_score(texts)

    # Severity = Volume x Speed x Negativity
    severity = volume * speed * negativity

    # Confidence = combination of volume + wording similarity
    volume_conf = min(volume / 20, 1.0)
    consistency = wording_similarity(texts)
    confidence = round((volume_conf + consistency) / 2, 2)

    # Uncertainty flag
    needs_review = (volume < 5 or confidence < 0.4 or negativity < 0.3)

    # Human-readable explanation
    if topic_df["topic"].iloc[0] == "service_outage":
        why = "Spike in outage-related complaints may indicate a live service disruption."
    elif topic_df["topic"].iloc[0] == "scam_rumor":
        why = "Repeated scam mentions could harm customer trust if unaddressed."
    elif topic_df["topic"].iloc[0] == "brand_sentiment":
        why = "Increasing negative sentiment about the brand may affect reputation."
    else:
        why = "Sudden shift in public sentiment detected."

    return {
        "topic": topic_df["topic"].iloc[0],
        "severity": round(severity, 2),
        "confidence": confidence,
        "why_it_matters": why,
        "needs_review": needs_review
    }

# ---- Process each topic ----
signal_objects = []
for topic in df["topic"].unique():
    topic_df = df[df["topic"] == topic]
    signal = compute_signal(topic_df)
    signal_objects.append(signal)

# ---- Save output for dashboard ----
with open("data/processed/signals.json", "w") as f:
    json.dump(signal_objects, f, indent=2)

print("✅ Scoring complete! signals.json ready in data/processed/")
