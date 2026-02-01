# src/sentiment_ml.py
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV


def train_sentiment_model(train_csv_path: str, text_col="text", label_col="label"):
    """
    Strong sentiment model for short public chatter.

    - Word TF-IDF (1-2 grams) captures phrases like "not happy", "down again"
    - Char TF-IDF captures noisy tokens / variants ("can't", "otp", misspellings)
    - LinearSVC works well for text
    - Calibration provides usable probabilities for confidence scoring
    """
    train_df = pd.read_csv(train_csv_path)

    if text_col not in train_df.columns or label_col not in train_df.columns:
        raise ValueError(
            f"Training CSV must include columns '{text_col}' and '{label_col}'. "
            f"Found: {list(train_df.columns)}"
        )

    X = train_df[text_col].fillna("").astype(str)
    y = train_df[label_col].astype(str)

    word_vec = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95
    )

    char_vec = TfidfVectorizer(
        lowercase=True,
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
        max_df=0.95
    )

    features = FeatureUnion([
        ("word_tfidf", word_vec),
        ("char_tfidf", char_vec)
    ])

    base_clf = LinearSVC(class_weight="balanced")

    # ✅ sklearn compatibility: use estimator= (new versions)
    clf = CalibratedClassifierCV(estimator=base_clf, method="sigmoid", cv=3)

    model = Pipeline([
        ("features", features),
        ("clf", clf)
    ])

    model.fit(X, y)
    return model


def predict_sentiment(model, texts: pd.Series, neutral_threshold: float = 0.70):
    """
    Predict sentiment + confidence.
    If confidence < neutral_threshold, return neutral (optional but demo-friendly).

    Returns:
      sentiment_ml, sentiment_confidence
    """
    X = texts.fillna("").astype(str)

    proba = model.predict_proba(X)
    classes = model.classes_

    best_idx = np.argmax(proba, axis=1)
    best_label = classes[best_idx]
    best_conf = proba[np.arange(len(X)), best_idx]

    best_label = np.where(best_conf < neutral_threshold, "neutral", best_label)

    return pd.DataFrame({
        "sentiment_ml": best_label,
        "sentiment_confidence": best_conf
    })
