# src/clustering.py
import os
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from sentiment_ml import train_sentiment_model, predict_sentiment


def map_cluster_to_scenario(cluster_keywords_or_label):
    """
    Map a cluster (by its keywords list OR label string) to a required scenario:
      - "service/incident"
      - "scam/rumor"
      - "brand sentiment"
      - "other/unclear"
    """
    if isinstance(cluster_keywords_or_label, (list, tuple)):
        text = " ".join(cluster_keywords_or_label).lower()
    else:
        text = str(cluster_keywords_or_label).lower()

    service_terms = [
        "outage", "down", "login", "log in", "access", "online banking", "slow", "error",
        "failed", "declined", "payment", "card payment", "atm", "withdrawal", "app", "glitch"
    ]
    scam_terms = [
        "otp", "phishing", "scam", "fraud", "fake", "fake call", "fake support", "sms",
        "skimming", "impersonat", "spoof"
    ]
    sentiment_terms = [
        "good", "great", "helpful", "smooth", "positive", "bad", "poor", "terrible",
        "customer service", "fees", "charges", "benefits", "rewards", "experience"
    ]
    rumor_terms = [
        "rumor", "unverified", "claim", "unclear", "breach", "data breach", "frozen", "freeze"
    ]

    def score(terms):
        return sum(1 for t in terms if t in text)

    service_score = score(service_terms)
    scam_score = score(scam_terms)
    sentiment_score = score(sentiment_terms)
    rumor_score = score(rumor_terms)

    if scam_score >= 1 and scam_score >= service_score:
        return "scam/rumor"

    if service_score >= 1 and service_score >= sentiment_score:
        return "service/incident"

    if sentiment_score >= 1 or (rumor_score >= 1 and service_score == 0 and scam_score == 0):
        return "brand sentiment"

    return "other/unclear"


def compute_certainty(df: pd.DataFrame, channel_col: str = "channel", likes_col: str = "likes") -> pd.Series:
    """
    Certainty score: 1..10
    - Channel baseline: news_comment higher, social_media lower
    - Social media likes: more likes -> higher certainty (signal strength, not truth)
    - Cluster volume boost: more mentions in same cluster -> slightly higher certainty
    """
    channel_base = {
        "news_comment": 8.0,
        "review_site": 6.5,
        "app_store": 6.0,
        "public_forum": 5.5,
        "community_chat": 5.0,
        "social_media": 3.5,
    }

    base = df[channel_col].map(channel_base).fillna(5.0).astype(float)

    likes = pd.to_numeric(df.get(likes_col, pd.Series([0] * len(df))), errors="coerce").fillna(0).astype(float)

    likes_boost = (np.log10(likes + 1) / np.log10(5000 + 1)) * 3.0
    likes_boost = likes_boost.clip(0, 3.0)

    likes_boost = np.where(df[channel_col].astype(str).str.lower() == "social_media", likes_boost, 0.0)

    if "cluster_id" in df.columns:
        cluster_counts = df["cluster_id"].value_counts()
        max_count = max(cluster_counts.max(), 1)
        volume_boost = df["cluster_id"].map(lambda c: (cluster_counts.get(c, 1) / max_count) * 1.5).astype(float)
    else:
        volume_boost = 0.0

    certainty = base + likes_boost + volume_boost
    certainty = np.clip(np.round(certainty), 1, 10).astype(int)
    return pd.Series(certainty, index=df.index, name="certainty")


def cluster_topics(
    data_path="data/synthesised_posts",
    text_col="text",
    channel_col="channel",
    likes_col="likes",
    top_k_words=6,
    candidate_k=(5, 6, 7, 8),
    save_output=True,
    sentiment_train_path="data/sentiment_train.csv"
):
    """
    Topic clustering (explainable) using TF-IDF + KMeans.
    Reads: data_path(.csv) or a directory containing a CSV.

    Adds:
      - cluster_id
      - cluster_label
      - scenario_from_cluster
      - certainty (1..10)
      - sentiment (ML)
      - sentiment_confidence (ML prob)

    Returns:
      df, cluster_summary, cluster_keywords, cluster_labels, out_path
    """
    # Resolve file path
    if os.path.isdir(data_path):
        csvs = [f for f in os.listdir(data_path) if f.lower().endswith(".csv")]
        if not csvs:
            raise FileNotFoundError(f"No CSV found in directory: {data_path}")
        file_path = os.path.join(data_path, csvs[0])
    else:
        file_path = data_path if data_path.lower().endswith(".csv") else data_path + ".csv"

    df = pd.read_csv(file_path)

    if text_col not in df.columns:
        raise ValueError(f"Missing '{text_col}' column. Found columns: {list(df.columns)}")
    if channel_col not in df.columns:
        raise ValueError(f"Missing '{channel_col}' column. Found columns: {list(df.columns)}")

    texts = df[text_col].fillna("").astype(str)

    # TF-IDF
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9
    )
    X = vectorizer.fit_transform(texts)

    # Auto-pick K via silhouette
    best_k, best_sil = None, -1
    for k in candidate_k:
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels)
        if sil > best_sil:
            best_k, best_sil = k, sil

    print(f"[Clustering] Auto-selected k={best_k} (silhouette={best_sil:.3f})")

    # Fit final model
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init="auto")
    df["cluster_id"] = kmeans.fit_predict(X)

    # Label clusters using top keywords
    feature_names = np.array(vectorizer.get_feature_names_out())
    centers = kmeans.cluster_centers_

    cluster_keywords = {}
    cluster_labels = {}

    for cid in range(best_k):
        top_idx = centers[cid].argsort()[::-1][:top_k_words]
        keywords = feature_names[top_idx].tolist()
        cluster_keywords[cid] = keywords
        cluster_labels[cid] = ", ".join(keywords[:3])

    df["cluster_label"] = df["cluster_id"].map(cluster_labels)

    # Scenario mapping
    df["scenario_from_cluster"] = df["cluster_id"].map(
        lambda cid: map_cluster_to_scenario(cluster_keywords[cid])
    )

    # Certainty score (1..10)
    df["certainty"] = compute_certainty(df, channel_col=channel_col, likes_col=likes_col)

    # -----------------------
    # ML Sentiment (train -> predict)
    # -----------------------
    if not os.path.exists(sentiment_train_path):
        raise FileNotFoundError(
            f"Sentiment training file not found: {sentiment_train_path}\n"
            f"Create it with columns: '{text_col}' (or 'text') and 'label'."
        )

    model = train_sentiment_model(sentiment_train_path, text_col=text_col, label_col="label")
    sent_out = predict_sentiment(model, df[text_col])

    df["sentiment"] = sent_out["sentiment_ml"]
    df["sentiment_confidence"] = sent_out["sentiment_confidence"]

    # Summary table
    cluster_summary = (
        df.groupby(["cluster_id", "cluster_label"])
          .size()
          .reset_index(name="count")
          .sort_values("count", ascending=False)
    )

    # Save
    out_path = os.path.splitext(file_path)[0] + "_clustered.csv"
    if save_output:
        df.to_csv(out_path, index=False)
        print(f"Saved clustered output to: {out_path}")

    return df, cluster_summary, cluster_keywords, cluster_labels, out_path


def main():
    df, cluster_summary, cluster_keywords, cluster_labels, out_path = cluster_topics()

    print("\n=== Cluster Summary ===")
    print(cluster_summary.to_string(index=False))

    print("\n=== Cluster Keywords (Explainability) ===")
    for cid in sorted(cluster_keywords.keys()):
        print(f"Cluster {cid} ({cluster_labels[cid]}): {cluster_keywords[cid]}")

    print("\n=== Sentiment counts ===")
    print(df["sentiment"].value_counts(dropna=False))

    print("\n=== Certainty (1..10) quick stats ===")
    print(df["certainty"].describe())

    print("\n=== Sentiment confidence quick stats ===")
    print(df["sentiment_confidence"].describe())

    print(f"\nSaved (with scenario + certainty + ML sentiment): {out_path}")


if __name__ == "__main__":
    main()
