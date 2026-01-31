# src/risk_scoring.py
import json
import math
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd


# -------------------------
# Config (tweakable knobs)
# -------------------------
@dataclass
class RiskConfig:
    # Time windows for momentum
    recent_hours: int = 2
    prev_hours: int = 6

    # Priority thresholds
    high_threshold: int = 70
    medium_threshold: int = 40

    # If confidence is below this and risk is high -> require human review
    review_conf_threshold: float = 0.65

    # Severity weights by scenario
    severity_weights: Dict[str, float] = None

    # Channel amplification weights (signal strength proxy)
    channel_amp: Dict[str, float] = None


def default_config() -> RiskConfig:
    return RiskConfig(
        severity_weights={
            "scam/rumor": 45.0,
            "service/incident": 40.0,
            "misinformation/false claims": 32.0,
            "brand sentiment": 22.0,
            "other/unclear": 10.0,
        },
        channel_amp={
            "news_comment": 1.25,
            "review_site": 1.10,
            "app_store": 1.05,
            "public_forum": 1.00,
            "community_chat": 0.95,
            "social_media": 0.90,
        },
    )


# -------------------------
# Helpers
# -------------------------
def safe_to_datetime(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", utc=True)


def likes_factor(likes: float) -> float:
    """
    Convert likes -> 0..1 amplification factor with log scaling.
    5 likes ~ small bump, 10k likes ~ near max.
    """
    if likes is None or (isinstance(likes, float) and np.isnan(likes)):
        return 0.0
    likes = max(float(likes), 0.0)
    # log10(1..10000) -> 0..4 ; scale to 0..1
    return float(min(math.log10(likes + 1.0) / 4.0, 1.0))


def compute_confidence(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    Combine:
      - certainty: 1..10 (your proxy signal confidence)
      - sentiment_confidence: 0..1 (model probability-ish)
    into:
      - confidence: 0..1
      - uncertainty: 0..1
    """
    certainty = pd.to_numeric(df.get("certainty", 5), errors="coerce").fillna(5).clip(1, 10) / 10.0
    sent_conf = pd.to_numeric(df.get("sentiment_confidence", 0.6), errors="coerce").fillna(0.6).clip(0, 1)

    confidence = (0.6 * certainty) + (0.4 * sent_conf)
    uncertainty = 1.0 - confidence
    return confidence.rename("confidence"), uncertainty.rename("uncertainty")


def compute_momentum_features(df: pd.DataFrame, now_utc: pd.Timestamp, cfg: RiskConfig) -> pd.DataFrame:
    """
    Per-cluster volume + growth:
      - recent_count: mentions in last recent_hours
      - prev_count: mentions in the window before that (prev_hours)
      - growth_ratio: (recent_rate)/(prev_rate) with smoothing
    """
    ts = safe_to_datetime(df["timestamp"])
    df = df.copy()
    df["_ts"] = ts

    recent_start = now_utc - pd.Timedelta(hours=cfg.recent_hours)
    prev_start = now_utc - pd.Timedelta(hours=cfg.prev_hours + cfg.recent_hours)
    prev_end = recent_start

    recent = df[(df["_ts"] >= recent_start) & (df["_ts"] <= now_utc)]
    prev = df[(df["_ts"] >= prev_start) & (df["_ts"] < prev_end)]

    recent_counts = recent["cluster_id"].value_counts()
    prev_counts = prev["cluster_id"].value_counts()

    # convert to per-hour rates with smoothing
    recent_rate = recent_counts / max(cfg.recent_hours, 1)
    prev_rate = prev_counts / max(cfg.prev_hours, 1)

    # Smoothing to avoid division by zero
    def growth(cid: int) -> float:
        r = float(recent_rate.get(cid, 0.0))
        p = float(prev_rate.get(cid, 0.0))
        return (r + 0.25) / (p + 0.25)

    df["recent_count"] = df["cluster_id"].map(lambda c: int(recent_counts.get(c, 0)))
    df["prev_count"] = df["cluster_id"].map(lambda c: int(prev_counts.get(c, 0)))
    df["growth_ratio"] = df["cluster_id"].map(growth)

    df.drop(columns=["_ts"], inplace=True)
    return df


# -------------------------
# Core risk scoring
# -------------------------
def compute_risk_score(df_in: pd.DataFrame, cfg: Optional[RiskConfig] = None, now_utc: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """
    Adds:
      - confidence, uncertainty
      - recent_count, prev_count, growth_ratio
      - risk_score (0..100)
      - priority (high/medium/low)
      - review_required (bool)
      - risk_drivers (string)
    """
    cfg = cfg or default_config()
    df = df_in.copy()

    # Validate minimum columns
    required = ["timestamp", "cluster_id", "cluster_label", "scenario_from_cluster", "channel"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {list(df.columns)}")

    # Establish "now" from data if not provided
    ts = safe_to_datetime(df["timestamp"])
    now_utc = now_utc or ts.max()
    if pd.isna(now_utc):
        now_utc = pd.Timestamp.utcnow()

    # Confidence / uncertainty
    df["confidence"], df["uncertainty"] = compute_confidence(df)

    # Momentum features
    df = compute_momentum_features(df, now_utc=now_utc, cfg=cfg)

    # Severity (scenario weights)
    sev_map = cfg.severity_weights or {}
    df["severity_points"] = df["scenario_from_cluster"].map(lambda s: float(sev_map.get(str(s).lower(), sev_map.get(s, 12.0))))

    # Sentiment impact (simple + explainable)
    # Negative chatter increases risk slightly; positive reduces slightly; neutral does nothing.
    sentiment = df.get("sentiment", "neutral").astype(str).str.lower()
    df["sentiment_points"] = np.select(
        [sentiment.eq("negative"), sentiment.eq("positive")],
        [8.0, -4.0],
        default=0.0
    )

    # Amplification (channel + likes)
    amp_map = cfg.channel_amp or {}
    ch = df["channel"].astype(str)
    ch_amp = ch.map(lambda c: float(amp_map.get(c, 1.0)))
    likes = pd.to_numeric(df.get("likes", np.nan), errors="coerce")

    # Convert to points 0..20
    # channel is a multiplier, likes adds 0..1, we combine
    df["amplification_points"] = (ch_amp * (0.4 + likes.fillna(0).map(likes_factor))) * 12.0
    df["amplification_points"] = df["amplification_points"].clip(0, 20)

    # Momentum points 0..25 based on recent volume + growth ratio
    # - growth_ratio ~1 means stable
    # - >2 means spiking
    growth = pd.to_numeric(df["growth_ratio"], errors="coerce").fillna(1.0)
    recent_count = pd.to_numeric(df["recent_count"], errors="coerce").fillna(0)

    # Normalize recent_count by max to keep stable across datasets
    max_recent = max(float(recent_count.max()), 1.0)
    vol_component = (recent_count / max_recent) * 10.0  # 0..10

    # growth component: log scale and cap
    growth_component = np.log2(growth.clip(lower=0.25, upper=8.0)) * 6.0  # ~ -12..18
    growth_component = np.clip(growth_component, 0, 15)  # focus on upward spikes only

    df["momentum_points"] = (vol_component + growth_component).clip(0, 25)

    # Raw risk score (linear, explainable) then clip to 0..100
    df["risk_score_raw"] = (
        df["severity_points"] +
        df["momentum_points"] +
        df["amplification_points"] +
        df["sentiment_points"]
    )

    # Confidence handling:
    # We do NOT hide risk when confidence is low. Instead:
    # - Keep risk_score as raw (clipped)
    # - Use review_required flag to enforce human oversight
    df["risk_score"] = df["risk_score_raw"].clip(0, 100).round().astype(int)

    # Priority buckets
    df["priority"] = np.select(
        [df["risk_score"] >= cfg.high_threshold, df["risk_score"] >= cfg.medium_threshold],
        ["high", "medium"],
        default="low"
    )

    # Human review gate (Responsible AI)
    df["review_required"] = (df["priority"].eq("high")) & (df["confidence"] < cfg.review_conf_threshold)

    # Explainability string
    df["risk_drivers"] = (
        "scenario=" + df["scenario_from_cluster"].astype(str) +
        "; severity=" + df["severity_points"].round(1).astype(str) +
        "; momentum=" + df["momentum_points"].round(1).astype(str) +
        "; amplification=" + df["amplification_points"].round(1).astype(str) +
        "; sentiment=" + df.get("sentiment", "neutral").astype(str) +
        "; certainty=" + df.get("certainty", "").astype(str) +
        "; confidence=" + df["confidence"].round(2).astype(str)
    )

    return df


# -------------------------
# Dashboard summary output
# -------------------------
def build_dashboard_summary(df: pd.DataFrame, now_utc: Optional[pd.Timestamp] = None, top_n: int = 8) -> Dict[str, Any]:
    ts = safe_to_datetime(df["timestamp"])
    now_utc = now_utc or ts.max()
    if pd.isna(now_utc):
        now_utc = pd.Timestamp.utcnow()

    # Top clusters by max risk in cluster (good for “Executive Insight Briefing”)
    cluster_rollup = (
        df.groupby(["cluster_id", "cluster_label", "scenario_from_cluster"])
          .agg(
              mentions=("cluster_id", "size"),
              avg_risk=("risk_score", "mean"),
              max_risk=("risk_score", "max"),
              avg_confidence=("confidence", "mean"),
              high_priority=("priority", lambda s: int((s == "high").sum()))
          )
          .reset_index()
          .sort_values(["max_risk", "mentions"], ascending=[False, False])
    )

    top_clusters = []
    for _, row in cluster_rollup.head(top_n).iterrows():
        cid = int(row["cluster_id"])
        examples = df[df["cluster_id"] == cid].sort_values("risk_score", ascending=False).head(3)
        top_clusters.append({
            "cluster_id": cid,
            "cluster_label": row["cluster_label"],
            "scenario": row["scenario_from_cluster"],
            "mentions": int(row["mentions"]),
            "avg_risk": float(round(row["avg_risk"], 1)),
            "max_risk": int(row["max_risk"]),
            "avg_confidence": float(round(row["avg_confidence"], 2)),
            "examples": examples["text"].astype(str).tolist() if "text" in df.columns else []
        })

    summary = {
        "generated_at_utc": str(now_utc),
        "totals": {
            "rows": int(len(df)),
            "priority_counts": df["priority"].value_counts(dropna=False).to_dict(),
            "scenario_counts": df["scenario_from_cluster"].value_counts(dropna=False).to_dict(),
            "sentiment_counts": df.get("sentiment", pd.Series(["neutral"] * len(df))).value_counts(dropna=False).to_dict(),
            "review_required_count": int(df["review_required"].sum()),
        },
        "top_clusters": top_clusters
    }
    return summary


def main(
    input_csv: str = "data/synthesised_posts_clustered.csv",
    out_csv: Optional[str] = None,
    out_json: str = "data/dashboard_summary.json"
):
    df = pd.read_csv(input_csv)

    cfg = default_config()
    df_scored = compute_risk_score(df, cfg=cfg)

    # Write enriched CSV
    out_csv = out_csv or (input_csv.replace(".csv", "_risk.csv"))
    df_scored.to_csv(out_csv, index=False)

    # Write dashboard summary JSON
    summary = build_dashboard_summary(df_scored)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Saved: {out_csv}")
    print(f"Saved: {out_json}")
    print("High priority:", int((df_scored["priority"] == "high").sum()))
    print("Review required:", int(df_scored["review_required"].sum()))


if __name__ == "__main__":
    main()
