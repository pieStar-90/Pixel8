# src/risk_scoring.py
import json
import math
import os
import re
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer


# -------------------------
# Config (tweakable knobs)
# -------------------------
@dataclass
class RiskConfig:
    # Time windows for momentum/trends
    recent_hours: int = 2
    prev_hours: int = 6

    # Priority thresholds (post-level)
    high_threshold: int = 70
    medium_threshold: int = 40

    # If confidence is below this and risk is high -> require human review
    review_conf_threshold: float = 0.65

    # Severity weights by topic/scenario (prefer topic_group)
    severity_weights: Dict[str, float] = None

    # Channel amplification weights (signal strength proxy)
    channel_amp: Dict[str, float] = None

    # ---- Issue/subtopic rollup rules ----
    # Issue priority thresholds (issue-level)
    issue_high_threshold: int = 70
    issue_medium_threshold: int = 40

    # Optional extra gate to stop “one-off” issues from appearing as high
    issue_min_mentions_for_high: int = 2

    # TF-IDF subtopic extraction
    subtopic_top_terms: int = 2          # 2 is enough for clean labels
    subtopic_ngram_range: Tuple[int, int] = (1, 2)
    subtopic_min_df: int = 2
    subtopic_max_df: float = 0.95
    subtopic_max_features: int = 6000

    # Local LLM labeling for subtopics (Ollama)
    enable_local_llm_labels: bool = True
    ollama_url: str = "http://localhost:11434/api/generate"
    ollama_model: str = "llama3.1:8b"
    subtopic_title_cache_path: str = "data/subtopic_titles.json"
    llm_label_top_n: int = 12  # only label top N issues


def default_config() -> RiskConfig:
    """
    3-class topic setup:
      - service_outage
      - scam_rumour
      - general_feedback

    Local LLM issue titles: ON by default.
    You can override model/url via env:
      OLLAMA_MODEL, OLLAMA_URL
    """
    cfg = RiskConfig(
        severity_weights={
            "scam_rumour": 45.0,
            "service_outage": 40.0,
            "general_feedback": 14.0,

            # fallback (if older cols exist)
            "scam/rumor": 45.0,
            "service/incident": 40.0,
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
        enable_local_llm_labels=True,
    )

    # optional env overrides
    cfg.ollama_model = os.getenv("OLLAMA_MODEL", cfg.ollama_model)
    cfg.ollama_url = os.getenv("OLLAMA_URL", cfg.ollama_url)

    return cfg


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
      - certainty: 1..10
      - sentiment_confidence: 0..1
      - topic_group_confidence: 0..1
    into:
      - confidence: 0..1
      - uncertainty: 0..1
    """
    certainty = (
        pd.to_numeric(df.get("certainty", 5), errors="coerce")
          .fillna(5).clip(1, 10) / 10.0
    )
    sent_conf = (
        pd.to_numeric(df.get("sentiment_confidence", 0.6), errors="coerce")
          .fillna(0.6).clip(0, 1)
    )
    topic_conf = (
        pd.to_numeric(df.get("topic_group_confidence", 0.6), errors="coerce")
          .fillna(0.6).clip(0, 1)
    )

    confidence = (0.5 * certainty) + (0.3 * sent_conf) + (0.2 * topic_conf)
    uncertainty = 1.0 - confidence
    return confidence.rename("confidence"), uncertainty.rename("uncertainty")


def compute_topic_momentum(df: pd.DataFrame, now_utc: pd.Timestamp, cfg: RiskConfig) -> pd.DataFrame:
    """
    Per-topic_group volume + growth:
      - recent_count: mentions in last recent_hours
      - prev_count: mentions in the window before that
      - growth_ratio: (recent_rate)/(prev_rate) with smoothing
    """
    df = df.copy()
    df["_ts"] = safe_to_datetime(df["timestamp"])

    recent_start = now_utc - pd.Timedelta(hours=cfg.recent_hours)
    prev_start = now_utc - pd.Timedelta(hours=cfg.prev_hours + cfg.recent_hours)
    prev_end = recent_start

    recent = df[(df["_ts"] >= recent_start) & (df["_ts"] <= now_utc)]
    prev = df[(df["_ts"] >= prev_start) & (df["_ts"] < prev_end)]

    recent_counts = recent["topic_group"].value_counts(dropna=False)
    prev_counts = prev["topic_group"].value_counts(dropna=False)

    recent_rate = recent_counts / max(cfg.recent_hours, 1)
    prev_rate = prev_counts / max(cfg.prev_hours, 1)

    def growth(key: str) -> float:
        r = float(recent_rate.get(key, 0.0))
        p = float(prev_rate.get(key, 0.0))
        return (r + 0.25) / (p + 0.25)

    df["recent_count"] = df["topic_group"].map(lambda k: int(recent_counts.get(k, 0)))
    df["prev_count"] = df["topic_group"].map(lambda k: int(prev_counts.get(k, 0)))
    df["growth_ratio"] = df["topic_group"].map(lambda k: float(growth(k)))

    df.drop(columns=["_ts"], inplace=True)
    return df


def get_severity_points(df: pd.DataFrame, cfg: RiskConfig) -> pd.Series:
    sev_map = cfg.severity_weights or {}
    tg = df.get("topic_group", "general_feedback").astype(str).str.strip()
    return tg.map(lambda x: float(sev_map.get(x, 12.0))).rename("severity_points")


def build_timeseries_6h_7d(df: pd.DataFrame, now_utc: pd.Timestamp) -> List[Dict[str, Any]]:
    """
    Returns a list of points every 6 hours for the last 7 days:
      [{ bucket_start_utc, service_outage, scam_rumour, negative_sentiment }, ...]
    """
    tmp = df.copy()
    tmp["_ts"] = safe_to_datetime(tmp["timestamp"])

    end = now_utc.floor("6H")
    start = end - pd.Timedelta(days=7)

    tmp = tmp[(tmp["_ts"] >= start) & (tmp["_ts"] <= end)].copy()
    tmp["bucket"] = tmp["_ts"].dt.floor("6H")

    buckets = pd.date_range(start=start, end=end, freq="6H", tz="UTC")

    service = tmp[tmp.get("topic_group", "") == "service_outage"].groupby("bucket").size()
    scam = tmp[tmp.get("topic_group", "") == "scam_rumour"].groupby("bucket").size()
    neg = tmp[tmp.get("sentiment", "").astype(str).str.lower() == "negative"].groupby("bucket").size()

    out: List[Dict[str, Any]] = []
    for b in buckets:
        out.append({
            "bucket_start_utc": str(b),
            "service_outage": int(service.get(b, 0)),
            "scam_rumour": int(scam.get(b, 0)),
            "negative_sentiment": int(neg.get(b, 0)),
        })
    return out


# -------------------------
# Subtopic / issue extraction (TF-IDF)
# -------------------------
_nonword = re.compile(r"[^a-z0-9_]+", re.IGNORECASE)


def _slug_term(t: str) -> str:
    t = t.strip().lower().replace(" ", "_")
    t = _nonword.sub("", t)
    return t[:40] if t else "misc"


def _top_terms_for_row(X_row, feature_names: np.ndarray, top_n: int) -> List[str]:
    """
    Return top_n terms for a sparse TF-IDF row.
    """
    if getattr(X_row, "nnz", 0) == 0:
        return ["misc"]
    data = X_row.data
    idx = X_row.indices
    order = np.argsort(data)[::-1]
    terms = [feature_names[idx[i]] for i in order[:top_n]]
    terms = [str(t).strip() for t in terms if str(t).strip()]
    return terms[:top_n] or ["misc"]


def add_subtopics_tfidf(
    df_in: pd.DataFrame,
    cfg: RiskConfig,
    text_col: str = "text",
    topic_col: str = "topic_group",
) -> pd.DataFrame:
    """
    Adds:
      - subtopic_key (stable grouping key)
      - subtopic_label (human-ish, TF-IDF based)

    Strategy: fit TF-IDF per topic_group for clean terms.
    """
    df = df_in.copy()

    if topic_col not in df.columns:
        df[topic_col] = "general_feedback"
    df[topic_col] = df[topic_col].fillna("general_feedback").astype(str)

    if text_col not in df.columns:
        df[text_col] = ""
    df[text_col] = df[text_col].fillna("").astype(str)

    subtopic_key = pd.Series([""] * len(df), index=df.index, dtype="object")
    subtopic_label = pd.Series([""] * len(df), index=df.index, dtype="object")

    for tg, idxs in df.groupby(topic_col).groups.items():
        idxs_list = list(idxs)
        texts = df.loc[idxs_list, text_col].astype(str).tolist()

        # Too few samples -> fallback
        if len(texts) < max(cfg.subtopic_min_df, 3):
            subtopic_key.loc[idxs_list] = f"{tg}:misc"
            subtopic_label.loc[idxs_list] = "misc"
            continue

        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=cfg.subtopic_ngram_range,
            min_df=cfg.subtopic_min_df,
            max_df=cfg.subtopic_max_df,
            max_features=cfg.subtopic_max_features,
        )

        X = vectorizer.fit_transform(texts)
        feature_names = np.array(vectorizer.get_feature_names_out())

        for j, row_idx in enumerate(idxs_list):
            terms = _top_terms_for_row(X[j], feature_names, cfg.subtopic_top_terms)
            slugs = [_slug_term(t) for t in terms]
            key = f"{tg}:" + "|".join(slugs)
            nice = " / ".join([s.replace("_", " ") for s in slugs])
            subtopic_key.at[row_idx] = key
            subtopic_label.at[row_idx] = nice

    df["subtopic_key"] = subtopic_key
    df["subtopic_label"] = subtopic_label
    return df


def _trend_counts_generic(
    df: pd.DataFrame,
    now_utc: pd.Timestamp,
    cfg: RiskConfig,
    col: str,
    allowed_values: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Returns recent vs previous counts and growth_ratio for each value in `col`.
    """
    tmp = df.copy()
    tmp["_ts"] = safe_to_datetime(tmp["timestamp"])

    recent_start = now_utc - pd.Timedelta(hours=cfg.recent_hours)
    prev_start = now_utc - pd.Timedelta(hours=cfg.prev_hours + cfg.recent_hours)
    prev_end = recent_start

    recent = tmp[(tmp["_ts"] >= recent_start) & (tmp["_ts"] <= now_utc)]
    prev = tmp[(tmp["_ts"] >= prev_start) & (tmp["_ts"] < prev_end)]

    recent_counts = recent[col].astype(str).value_counts()
    prev_counts = prev[col].astype(str).value_counts()

    if allowed_values is None:
        keys = sorted(set(recent_counts.index.tolist()) | set(prev_counts.index.tolist()))
    else:
        keys = allowed_values

    out: Dict[str, Any] = {}
    for k in keys:
        r = int(recent_counts.get(k, 0))
        p = int(prev_counts.get(k, 0))
        r_rate = r / max(cfg.recent_hours, 1)
        p_rate = p / max(cfg.prev_hours, 1)
        growth = (r_rate + 0.25) / (p_rate + 0.25)
        out[str(k)] = {"recent": r, "previous": p, "growth_ratio": round(float(growth), 2)}

    tmp.drop(columns=["_ts"], inplace=True)
    return out


def _dominant_value(series: pd.Series) -> str:
    """
    Most common value in a series, with stable tie-break:
      - highest count
      - then alphabetical by value
    """
    if series is None or len(series) == 0:
        return "unknown"
    s = series.dropna().astype(str).str.strip()
    if s.empty:
        return "unknown"
    vc = s.value_counts()
    if vc.empty:
        return "unknown"
    top_count = int(vc.iloc[0])
    tied = vc[vc == top_count].index.astype(str).tolist()
    tied.sort()
    return tied[0] if tied else str(vc.index[0])


# -------------------------
# Local LLM labels for top issues (Ollama)
# -------------------------
def _load_title_cache(path: str) -> Dict[str, str]:
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
                if isinstance(obj, dict):
                    return {str(k): str(v) for k, v in obj.items()}
    except Exception:
        pass
    return {}


def _save_title_cache(path: str, cache: Dict[str, str]) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _try_ollama_label_issue(
    topic_group: str,
    keywords_label: str,
    examples: List[str],
    cfg: RiskConfig,
) -> Optional[str]:
    """
    Best-effort local label. If Ollama isn't running, raise a clear error.
    """
    try:
        import requests  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "Local LLM labeling is enabled but 'requests' is not installed.\n"
            "Fix: pip install requests"
        ) from e

    import requests

    prompt = f"""
You name issue categories for a bank risk dashboard.

Return ONLY a short title (3–7 words). No quotes. No extra text.

Constraints:
- Formal, bank-grade, neutral tone.
- Use a noun phrase (e.g., "Payment Transaction Failures"), not conversational phrasing.
- Avoid: trying, feels, happening, lot, like.
- No slashes, no filler, no emojis.
- Use ONLY the provided keywords + examples; do not speculate beyond them.
- No brand names.

Topic group: {topic_group}
Keywords label: {keywords_label}

Examples:
- {examples[0] if len(examples) > 0 else ""}
- {examples[1] if len(examples) > 1 else ""}
- {examples[2] if len(examples) > 2 else ""}
""".strip()

    try:
        r = requests.post(
            cfg.ollama_url,
            json={
                "model": cfg.ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.2},
            },
            timeout=45,
        )
        r.raise_for_status()
        out = (r.json().get("response") or "").strip()
    except requests.RequestException as e:
        raise RuntimeError(
            "Local LLM labeling is enabled but Ollama isn't reachable.\n"
            f"Tried: {cfg.ollama_url} (model={cfg.ollama_model})\n"
            "Fix:\n"
            "  - ensure Ollama is running (it usually auto-starts)\n"
            "  - verify with: curl http://127.0.0.1:11434/api/tags\n"
            "  - ensure model exists: ollama pull llama3.1:8b (or set OLLAMA_MODEL)\n"
        ) from e

    out = out.replace("\n", " ").strip()
    out = " ".join(out.split())           # normalize spaces
    out = out.strip().strip('"').strip("'").strip()

    # reject slashes and overly long titles
    if "/" in out:
        return None

    words = out.split()
    if len(words) < 3 or len(words) > 7:
        return None

    banned = {"trying", "feels", "happening", "lot", "like"}
    if any(w.lower().strip(".,!?") in banned for w in words):
        return None


    # basic guardrails
    if len(out) < 3 or len(out) > 80:
        return None

    # remove accidental quotes
    out = out.strip().strip('"').strip("'").strip()
    return out or None


# -------------------------
# Core risk scoring
# -------------------------
def compute_risk_score(
    df_in: pd.DataFrame,
    cfg: Optional[RiskConfig] = None,
    now_utc: Optional[pd.Timestamp] = None
) -> pd.DataFrame:
    """
    Adds:
      - confidence, uncertainty
      - recent_count, prev_count, growth_ratio (topic_group only)
      - risk_score (0..100)
      - priority (high/medium/low)
      - review_required (bool)
      - risk_drivers (string)
    """
    cfg = cfg or default_config()
    df = df_in.copy()

    required = ["timestamp", "channel"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {list(df.columns)}")

    if "topic_group" not in df.columns:
        df["topic_group"] = "general_feedback"

    ts = safe_to_datetime(df["timestamp"])
    now_utc = now_utc or ts.max()
    if pd.isna(now_utc):
        now_utc = pd.Timestamp.utcnow()

    df["confidence"], df["uncertainty"] = compute_confidence(df)
    df = compute_topic_momentum(df, now_utc=now_utc, cfg=cfg)
    df["severity_points"] = get_severity_points(df, cfg)

    sentiment = df.get("sentiment", "neutral").astype(str).str.lower()
    df["sentiment_points"] = np.select(
        [sentiment.eq("negative"), sentiment.eq("positive")],
        [8.0, -4.0],
        default=0.0
    )

    amp_map = cfg.channel_amp or {}
    ch_amp = df["channel"].astype(str).map(lambda c: float(amp_map.get(c, 1.0)))
    likes = pd.to_numeric(df.get("likes", np.nan), errors="coerce")

    df["amplification_points"] = (ch_amp * (0.4 + likes.fillna(0).map(likes_factor))) * 12.0
    df["amplification_points"] = df["amplification_points"].clip(0, 20)

    growth = pd.to_numeric(df["growth_ratio"], errors="coerce").fillna(1.0)
    recent_count = pd.to_numeric(df["recent_count"], errors="coerce").fillna(0)

    max_recent = max(float(recent_count.max()), 1.0)
    vol_component = (recent_count / max_recent) * 10.0

    growth_component = np.log2(growth.clip(lower=0.25, upper=8.0)) * 6.0
    growth_component = np.clip(growth_component, 0, 15)

    df["momentum_points"] = (vol_component + growth_component).clip(0, 25)

    df["risk_score_raw"] = (
        df["severity_points"] +
        df["momentum_points"] +
        df["amplification_points"] +
        df["sentiment_points"]
    )

    df["risk_score"] = df["risk_score_raw"].clip(0, 100).round().astype(int)

    df["priority"] = np.select(
        [df["risk_score"] >= cfg.high_threshold, df["risk_score"] >= cfg.medium_threshold],
        ["high", "medium"],
        default="low"
    )

    df["review_required"] = (df["priority"].eq("high")) & (df["confidence"] < cfg.review_conf_threshold)

    df["risk_drivers"] = (
        "topic=" + df["topic_group"].astype(str) +
        "; severity=" + df["severity_points"].round(1).astype(str) +
        "; momentum=" + df["momentum_points"].round(1).astype(str) +
        "; amplification=" + df["amplification_points"].round(1).astype(str) +
        "; sentiment=" + df.get("sentiment", "neutral").astype(str) +
        "; certainty=" + df.get("certainty", "").astype(str) +
        "; confidence=" + df["confidence"].round(2).astype(str)
    )

    df["momentum_group_col"] = "topic_group"
    return df


# -------------------------
# Dashboard summary output (Option A: NO KMeans)
# -------------------------
def build_dashboard_summary(
    df: pd.DataFrame,
    cfg: Optional[RiskConfig] = None,
    now_utc: Optional[pd.Timestamp] = None,
    top_n: int = 8,
    top_posts_n: int = 10
) -> Dict[str, Any]:
    """
    JSON contract includes:
      - totals: post-level counts + issue-level counts
      - trends: topic_group + sentiment
      - timeseries_6h_7d
      - top_topics
      - top_issues + issue_rollup (includes dominant_channel)
      - top_posts
      - llm_issue_titles_enabled
    """
    cfg = cfg or default_config()

    ts = safe_to_datetime(df["timestamp"])
    now_utc = now_utc or ts.max()
    if pd.isna(now_utc):
        now_utc = pd.Timestamp.utcnow()

    df = df.copy()
    if "topic_group" not in df.columns:
        df["topic_group"] = "general_feedback"
    if "sentiment" not in df.columns:
        df["sentiment"] = "neutral"
    if "text" not in df.columns:
        df["text"] = ""
    if "channel" not in df.columns:
        df["channel"] = "unknown"

    # -----------------------
    # Subtopics (issue keys) + rollups
    # -----------------------
    df = add_subtopics_tfidf(df, cfg=cfg, text_col="text", topic_col="topic_group")

    # trends (topic + sentiment)
    topic_trends = _trend_counts_generic(
        df, now_utc, cfg, "topic_group",
        allowed_values=["service_outage", "scam_rumour", "general_feedback"]
    )
    sentiment_trends = _trend_counts_generic(
        df, now_utc, cfg, "sentiment",
        allowed_values=["negative", "neutral", "positive"]
    )

    # issue trends by subtopic_key
    issue_trends = _trend_counts_generic(df, now_utc, cfg, "subtopic_key")

    # issue rollup (NOW includes dominant_channel)
    issue_rollup_df = (
        df.groupby(["subtopic_key", "topic_group", "subtopic_label"])
          .agg(
              mentions=("subtopic_key", "size"),
              avg_risk=("risk_score", "mean"),
              max_risk=("risk_score", "max"),
              avg_confidence=("confidence", "mean"),
              high_posts=("priority", lambda s: int((s == "high").sum())),
              medium_posts=("priority", lambda s: int((s == "medium").sum())),
              dominant_channel=("channel", _dominant_value),
          )
          .reset_index()
    )

    issue_rollup_df["recent_mentions"] = issue_rollup_df["subtopic_key"].map(
        lambda k: issue_trends.get(str(k), {}).get("recent", 0)
    )
    issue_rollup_df["previous_mentions"] = issue_rollup_df["subtopic_key"].map(
        lambda k: issue_trends.get(str(k), {}).get("previous", 0)
    )
    issue_rollup_df["growth_ratio"] = issue_rollup_df["subtopic_key"].map(
        lambda k: issue_trends.get(str(k), {}).get("growth_ratio", 1.0)
    )

    def issue_priority(row) -> str:
        max_r = float(row["max_risk"])
        mentions = int(row["mentions"])
        if max_r >= cfg.issue_high_threshold and mentions >= cfg.issue_min_mentions_for_high:
            return "high"
        if max_r >= cfg.issue_medium_threshold:
            return "medium"
        return "low"

    issue_rollup_df["issue_priority"] = issue_rollup_df.apply(issue_priority, axis=1)

    # rank issues for display
    pri_rank = {"high": 0, "medium": 1, "low": 2}
    issue_rollup_df["_pri_rank"] = issue_rollup_df["issue_priority"].map(lambda x: pri_rank.get(str(x), 9))
    issue_rollup_df = issue_rollup_df.sort_values(
        ["_pri_rank", "max_risk", "recent_mentions", "mentions"],
        ascending=[True, False, False, False]
    ).drop(columns=["_pri_rank"])

    # -----------------------
    # Local LLM labeling for top issues (Ollama) + cache
    # -----------------------
    title_cache: Dict[str, str] = _load_title_cache(cfg.subtopic_title_cache_path)

    if cfg.enable_local_llm_labels:
        top_issue_keys = issue_rollup_df.head(cfg.llm_label_top_n)["subtopic_key"].astype(str).tolist()

        for k in top_issue_keys:
            if k in title_cache:
                continue

            subset = df[df["subtopic_key"].astype(str) == k].sort_values("risk_score", ascending=False).head(3)
            examples = subset["text"].astype(str).tolist()

            row = issue_rollup_df[issue_rollup_df["subtopic_key"].astype(str) == k].head(1)
            if row.empty:
                continue

            topic_group = str(row.iloc[0]["topic_group"])
            keywords_label = str(row.iloc[0]["subtopic_label"])

            title = _try_ollama_label_issue(
                topic_group=topic_group,
                keywords_label=keywords_label,
                examples=examples,
                cfg=cfg,
            )
            if title:
                title_cache[k] = title

        _save_title_cache(cfg.subtopic_title_cache_path, title_cache)

    def best_issue_title(k: str, fallback: str) -> str:
        if cfg.enable_local_llm_labels and str(k) in title_cache:
            return title_cache[str(k)]
        return fallback

    issue_rollup_df["issue_title"] = issue_rollup_df.apply(
        lambda r: best_issue_title(str(r["subtopic_key"]), str(r["subtopic_label"])),
        axis=1
    )

    high_priority_issues = int((issue_rollup_df["issue_priority"] == "high").sum())
    medium_priority_issues = int((issue_rollup_df["issue_priority"] == "medium").sum())

    # -----------------------
    # Roll up by topic_group (existing)
    # -----------------------
    rollup = (
        df.groupby(["topic_group"])
          .agg(
              mentions=("topic_group", "size"),
              avg_risk=("risk_score", "mean"),
              max_risk=("risk_score", "max"),
              avg_confidence=("confidence", "mean"),
              high_priority=("priority", lambda s: int((s == "high").sum())),
              medium_priority=("priority", lambda s: int((s == "medium").sum())),
          )
          .reset_index()
          .sort_values(["max_risk", "mentions"], ascending=[False, False])
    )

    rollup["recent_mentions"] = rollup["topic_group"].map(lambda tg: topic_trends.get(str(tg), {}).get("recent", 0))
    rollup["previous_mentions"] = rollup["topic_group"].map(lambda tg: topic_trends.get(str(tg), {}).get("previous", 0))
    rollup["growth_ratio"] = rollup["topic_group"].map(lambda tg: topic_trends.get(str(tg), {}).get("growth_ratio", 1.0))

    top_topics: List[Dict[str, Any]] = []
    for _, row in rollup.head(top_n).iterrows():
        tg = row["topic_group"]
        examples_df = df[df["topic_group"] == tg].sort_values("risk_score", ascending=False).head(3)
        top_topics.append({
            "topic_group": tg,
            "mentions": int(row["mentions"]),
            "recent_mentions": int(row["recent_mentions"]),
            "previous_mentions": int(row["previous_mentions"]),
            "growth_ratio": float(row["growth_ratio"]),
            "avg_risk": float(round(row["avg_risk"], 1)),
            "max_risk": int(row["max_risk"]),
            "avg_confidence": float(round(row["avg_confidence"], 2)),
            "high_priority": int(row["high_priority"]),
            "medium_priority": int(row["medium_priority"]),
            "examples": examples_df["text"].astype(str).tolist() if "text" in df.columns else [],
        })

    # top posts feed (still used elsewhere)
    cols = [
        "timestamp", "channel", "likes", "text",
        "topic_group", "topic_group_confidence",
        "subtopic_key", "subtopic_label",
        "sentiment", "sentiment_confidence",
        "certainty", "confidence", "uncertainty",
        "risk_score", "priority", "review_required",
        "risk_drivers"
    ]
    available_cols = [c for c in cols if c in df.columns]
    top_posts_df = df.sort_values("risk_score", ascending=False).head(top_posts_n)
    top_posts = top_posts_df[available_cols].to_dict(orient="records")

    # top issues list (NOW includes dominant_channel)
    top_issues: List[Dict[str, Any]] = []
    for _, row in issue_rollup_df.head(top_n).iterrows():
        k = str(row["subtopic_key"])
        examples_df = df[df["subtopic_key"].astype(str) == k].sort_values("risk_score", ascending=False).head(3)
        top_issues.append({
            "subtopic_key": k,
            "issue_title": str(row["issue_title"]),
            "topic_group": str(row["topic_group"]),
            "subtopic_label": str(row["subtopic_label"]),  # explainable fallback
            "issue_priority": str(row["issue_priority"]),
            "dominant_channel": str(row.get("dominant_channel", "unknown")),
            "mentions": int(row["mentions"]),
            "recent_mentions": int(row["recent_mentions"]),
            "previous_mentions": int(row["previous_mentions"]),
            "growth_ratio": float(row["growth_ratio"]),
            "avg_risk": float(round(float(row["avg_risk"]), 1)),
            "max_risk": int(row["max_risk"]),
            "avg_confidence": float(round(float(row["avg_confidence"]), 2)),
            "high_posts": int(row["high_posts"]),
            "medium_posts": int(row["medium_posts"]),
            "examples": examples_df["text"].astype(str).tolist(),
        })

    # full issue rollup (NOW includes dominant_channel)
    issue_rollup = issue_rollup_df.head(60).to_dict(orient="records")

    # timeseries for chart
    timeseries_6h_7d = build_timeseries_6h_7d(df, now_utc=now_utc)

    summary = {
        "generated_at_utc": str(now_utc),
        "windows": {
            "recent_hours": int(cfg.recent_hours),
            "previous_hours": int(cfg.prev_hours),
        },
        "totals": {
            "rows": int(len(df)),
            "priority_counts": df["priority"].value_counts(dropna=False).to_dict(),
            "topic_group_counts": df["topic_group"].value_counts(dropna=False).to_dict(),
            "review_required_count": int(df["review_required"].sum()) if "review_required" in df.columns else 0,
            "high_priority_issues": high_priority_issues,
            "medium_priority_issues": medium_priority_issues,
        },
        "trends": {
            "topic_group": topic_trends,
            "sentiment": sentiment_trends,
        },
        "timeseries_6h_7d": timeseries_6h_7d,
        "top_topics": top_topics,
        "top_issues": top_issues,
        "issue_rollup": issue_rollup,
        "top_posts": top_posts,
        "llm_issue_titles_enabled": bool(cfg.enable_local_llm_labels),
        "ollama_model": cfg.ollama_model if cfg.enable_local_llm_labels else None,
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

    out_csv = out_csv or (input_csv.replace(".csv", "_risk.csv"))
    df_scored.to_csv(out_csv, index=False)

    summary = build_dashboard_summary(df_scored, cfg=cfg)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Saved: {out_csv}")
    print(f"Saved: {out_json}")

    print("High priority posts:", int((df_scored["priority"] == "high").sum()))
    print("Review required:", int(df_scored["review_required"].sum()))
    print("High priority issues:", int(summary["totals"]["high_priority_issues"]))
    print("Medium priority issues:", int(summary["totals"]["medium_priority_issues"]))
    if summary.get("llm_issue_titles_enabled"):
        print("LLM issue titles: ENABLED (Ollama)")
        print("Model:", summary.get("ollama_model"))
    else:
        print("LLM issue titles: DISABLED")


if __name__ == "__main__":
    main()
