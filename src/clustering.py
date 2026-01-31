# src/topic_clustering.py
import os
import re
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def map_cluster_to_scenario(cluster_keywords_or_label):
    """
    Map a cluster (by its keywords list OR label string) to a required scenario:
      - "service/incident"
      - "scam/rumor"
      - "brand sentiment"
      - "other/unclear"

    Explainable: checks for known indicator keywords in the cluster’s top TF-IDF terms
    (or in the label made from those terms).
    """
    # Accept list (keywords) or string (label)
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

    # Scam first
    if scam_score >= 1 and scam_score >= service_score:
        return "scam/rumor"

    # Service/incident
    if service_score >= 1 and service_score >= sentiment_score:
        return "service/incident"

    # Brand sentiment (and unverified rumor as sentiment risk when not service/scam)
    if sentiment_score >= 1 or (rumor_score >= 1 and service_score == 0 and scam_score == 0):
        return "brand sentiment"

    return "other/unclear"


def cluster_topics(
    data_path="data/synthesised_posts",
    text_col="chatter",
    top_k_words=6,
    candidate_k=(5, 6, 7, 8),
    save_output=True
):
    """
    Topic clustering (explainable) using TF-IDF + KMeans (English, ~200 rows).
    Reads: data_path(.csv) or a directory containing a CSV.

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

    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init="auto")
    cluster_id = kmeans.fit_predict(X)
    df["cluster_id"] = cluster_id

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
    # Runs the clustering pipeline and adds scenario mapping
    df, cluster_summary, cluster_keywords, cluster_labels, out_path = cluster_topics()

    df["scenario_from_cluster"] = df["cluster_id"].map(
        lambda cid: map_cluster_to_scenario(cluster_keywords[cid])
    )

    df.to_csv(out_path, index=False)
    print("\n=== Cluster Summary ===")
    print(cluster_summary.to_string(index=False))

    print("\n=== Cluster Keywords (Explainability) ===")
    for cid in sorted(cluster_keywords.keys()):
        print(f"Cluster {cid} ({cluster_labels[cid]}): {cluster_keywords[cid]}")

    print(f"\nSaved (with scenario mapping): {out_path}")


if __name__ == "__main__":
    main()
