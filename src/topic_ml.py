# src/topic_ml.py
import os
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

ALLOWED = ["general_feedback", "brand_sentiment", "service_outage", "scam_rumour"]


def _validate_labels(y: pd.Series):
    y = y.astype(str).str.strip()
    bad = sorted(set(y.unique()) - set(ALLOWED))
    if bad:
        raise ValueError(f"Unexpected labels: {bad}. Allowed: {ALLOWED}")
    return y


def train_topic_model(train_csv_path: str, text_col="text", label_col="label"):
    df = pd.read_csv(train_csv_path)
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"Training CSV must include '{text_col}' and '{label_col}'. Found: {list(df.columns)}")

    X = df[text_col].fillna("").astype(str)
    y = _validate_labels(df[label_col].fillna("").astype(str))

    # Word ngrams catch phrases ("log in", "data breach")
    word_vec = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )

    # Char ngrams help with noisy short tokens and variations ("otp", "cant", "log-in")
    char_vec = TfidfVectorizer(
        lowercase=True,
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_df=0.95
    )

    features = FeatureUnion([
        ("word", word_vec),
        ("char", char_vec)
    ])

    base = LinearSVC(class_weight="balanced")
    clf = CalibratedClassifierCV(estimator=base, method="sigmoid", cv=3)

    model = Pipeline([
        ("features", features),
        ("clf", clf)
    ])

    model.fit(X, y)
    return model


def evaluate_topic_model(train_csv_path: str, text_col="text", label_col="label", test_size=0.2, random_state=42):
    df = pd.read_csv(train_csv_path)
    X = df[text_col].fillna("").astype(str)
    y = _validate_labels(df[label_col].fillna("").astype(str))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    model = train_topic_model_from_xy(X_train, y_train)
    preds = model.predict(X_test)

    print("\n=== Classification Report ===")
    print(classification_report(y_test, preds, digits=3, labels=ALLOWED))

    print("\n=== Confusion Matrix (rows=true, cols=pred) ===")
    print(pd.DataFrame(confusion_matrix(y_test, preds, labels=ALLOWED), index=ALLOWED, columns=ALLOWED))

    return model


def train_topic_model_from_xy(X: pd.Series, y: pd.Series):
    X = X.fillna("").astype(str)
    y = _validate_labels(pd.Series(y))

    word_vec = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )

    char_vec = TfidfVectorizer(
        lowercase=True,
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_df=0.95
    )

    features = FeatureUnion([
        ("word", word_vec),
        ("char", char_vec)
    ])

    base = LinearSVC(class_weight="balanced")
    clf = CalibratedClassifierCV(estimator=base, method="sigmoid", cv=3)

    model = Pipeline([
        ("features", features),
        ("clf", clf)
    ])

    model.fit(X, y)
    return model


def predict_topic(model, texts: pd.Series):
    X = texts.fillna("").astype(str)
    proba = model.predict_proba(X)
    classes = model.classes_

    best_idx = np.argmax(proba, axis=1)
    best_label = classes[best_idx]
    best_conf = proba[np.arange(len(X)), best_idx]

    return pd.DataFrame({
        "topic_group": best_label,
        "topic_group_confidence": best_conf
    })
