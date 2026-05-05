"""
Q5 Kaggle baseline: bot vs. genuine Twitter user classification.

Competition: CS610 Assignment 1 Question 5 (2026)
Metric: AUC (Area Under ROC Curve) — submit PROBABILITIES, not hard labels.
Submission format: index,target  where target is a float in [0, 1].

Strategy:
- Cheap-to-compute features only (numerics, booleans, top-N langs,
  presence flags). No free-text NLP in v1.
- Train logistic regression and random forest, evaluate on a stratified
  80/20 holdout using AUC.
- Pick the higher-AUC model, refit on the full train set, predict test
  probabilities, write Kaggle-ready CSV.

Run from repo root or from q5_kaggle/ — paths are relative to this file.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DATA_DIR = REPO / "Assignment1" / "cs-610-assignment-1-question-5-2026"
SUB_DIR = HERE / "submissions"
SUB_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 2025

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")
print(f"train: {train.shape}  test: {test.shape}")
print(f"target balance: {train['target'].value_counts().to_dict()}")


# ---------------------------------------------------------------------------
# Feature engineering — simple, no leakage
# ---------------------------------------------------------------------------
TOP_LANGS = (
    train["lang"].value_counts().head(8).index.tolist()
)  # everything else -> "OTHER"


def featurize(df: pd.DataFrame) -> pd.DataFrame:
    """Lightweight feature pipeline applied to both train and test.

    Engineered features are kept simple so the baseline is reproducible
    and fast. Anything heavier (description NLP, screen_name patterns,
    location parsing) belongs in v2.
    """
    out = pd.DataFrame(index=df.index)

    # Numeric counts — pass through, model will scale them.
    for col in [
        "favourites_count",
        "followers_count",
        "friends_count",
        "statuses_count",
        "average_tweets_per_day",
        "account_age_days",
    ]:
        out[col] = df[col].astype(float)

    # Log1p versions of heavy-tailed counts (the raw scale is brutal).
    for col in [
        "favourites_count",
        "followers_count",
        "friends_count",
        "statuses_count",
    ]:
        out[f"log_{col}"] = np.log1p(out[col])

    # Booleans — already bool dtype in the CSV.
    for col in ["default_profile", "default_profile_image", "geo_enabled", "verified"]:
        out[col] = df[col].astype(int)

    # Presence flags (cheap, often informative).
    out["has_description"] = df["description"].notna().astype(int)
    out["has_location"] = (df["location"].notna() & (df["location"] != "unknown")).astype(int)
    out["has_url_bg"] = df["profile_background_image_url"].notna().astype(int)

    # description length (proxy for "did they bother to fill it in").
    out["desc_len"] = df["description"].fillna("").str.len()

    # screen_name characteristics (bots often have digit-heavy names).
    out["sn_len"] = df["screen_name"].fillna("").str.len()
    out["sn_digits"] = (
        df["screen_name"].fillna("").str.count(r"\d") / out["sn_len"].clip(lower=1)
    )

    # Lang as a coarse categorical (top 8 + OTHER + missing).
    lang = df["lang"].fillna("MISSING")
    out["lang"] = lang.where(lang.isin(TOP_LANGS), "OTHER")

    # Ratios — some classic bot heuristics.
    out["followers_per_friend"] = df["followers_count"] / df["friends_count"].clip(lower=1)
    out["statuses_per_day"] = df["statuses_count"] / df["account_age_days"].clip(lower=1)

    return out


X_train_full = featurize(train)
y_train = train["target"].values
X_test = featurize(test)

print(f"\nfeature matrix train: {X_train_full.shape}, test: {X_test.shape}")
print(f"feature cols: {X_train_full.columns.tolist()}")


# ---------------------------------------------------------------------------
# Preprocessing pipeline — different per model family
# ---------------------------------------------------------------------------
numeric_cols = [c for c in X_train_full.columns if c != "lang"]
categorical_cols = ["lang"]

# For LogReg: scale numerics, one-hot lang.
preproc_lr = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline(
                steps=[
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                ]
            ),
            numeric_cols,
        ),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_cols,
        ),
    ]
)

# For RF: just impute and one-hot. RF doesn't need scaling.
preproc_rf = ColumnTransformer(
    transformers=[
        (
            "num",
            SimpleImputer(strategy="median"),
            numeric_cols,
        ),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_cols,
        ),
    ]
)

models = {
    "logreg": Pipeline(
        steps=[
            ("preproc", preproc_lr),
            ("clf", LogisticRegression(max_iter=2000, C=1.0, random_state=RANDOM_STATE)),
        ]
    ),
    "rf": Pipeline(
        steps=[
            ("preproc", preproc_rf),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=None,
                    min_samples_leaf=2,
                    n_jobs=1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    ),
}

# ---------------------------------------------------------------------------
# Holdout evaluation — 80/20 stratified split for a quick verdict
# ---------------------------------------------------------------------------
Xa, Xb, ya, yb = train_test_split(
    X_train_full, y_train, test_size=0.2, stratify=y_train, random_state=RANDOM_STATE
)

scores = {}
for name, pipe in models.items():
    pipe.fit(Xa, ya)
    yb_pred = pipe.predict(Xb)
    yb_prob = pipe.predict_proba(Xb)[:, 1]
    auc = roc_auc_score(yb, yb_prob)
    f1 = f1_score(yb, yb_pred)
    scores[name] = {"auc": auc, "f1": f1}
    print(f"\n=== {name.upper()} (holdout) ===")
    print(f"  AUC: {auc:.4f}   <-- competition metric")
    print(f"  F1 : {f1:.4f}")
    print(classification_report(yb, yb_pred, digits=4))

# Pick the model with highest holdout AUC — that is the competition metric.
winner = max(scores, key=lambda k: scores[k]["auc"])
print(f"\nWinner by holdout AUC: {winner} (AUC={scores[winner]['auc']:.4f})")

# Refit the winning pipeline on the FULL training set before predicting test.
best = models[winner]
best.fit(X_train_full, y_train)

# ---------------------------------------------------------------------------
# Predict test PROBABILITIES, write submission
# ---------------------------------------------------------------------------
# Kaggle wants P(target=1), not hard labels — AUC needs the full ranking.
test_prob = best.predict_proba(X_test)[:, 1]

submission = pd.DataFrame({"index": test["index"].values, "target": test_prob})
out_path = SUB_DIR / f"baseline_{winner}.csv"
submission.to_csv(out_path, index=False, float_format="%.6f")
print(f"\nWrote submission: {out_path}  shape={submission.shape}")
print(submission.head(8))
print(f"\npredicted prob stats: "
      f"min={test_prob.min():.4f}  mean={test_prob.mean():.4f}  "
      f"max={test_prob.max():.4f}")
print(f"(train base rate for reference: {y_train.mean():.4f})")
