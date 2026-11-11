#!/usr/bin/env python3
"""Controlled Yelp snapshot-status classification study.

This is NOT a future-closure prediction experiment. Yelp's ``is_open`` label
has no closure timestamp, and some features are snapshot aggregates.

Usage (from repository root):
    python scripts/recorded_status_experiments.py
    python scripts/recorded_status_experiments.py --input data/business_features_stage1_final.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
COUNT_FEATURES = {
    "business_review_count", "reviews_2021", "reviews_first_6m_2021",
    "reviews_recent_6m_2021", "reviews_recent_3m_2021",
    "all_time_checkins", "tips_2021",
}
BUSINESS = ["business_stars", "business_review_count"]
VOLUME = [*BUSINESS, "reviews_2021"]
TEMPORAL = [*BUSINESS, "reviews_first_6m_2021", "reviews_recent_6m_2021"]
CUSTOMER = [*VOLUME, "all_time_checkins", "tips_2021"]
FULL = [*TEMPORAL, "all_time_checkins", "tips_2021",
        "avg_review_stars_2021", "rating_change_recent_vs_first_6m"]
FEATURE_GROUPS = {
    "01_business_only": BUSINESS,
    "02_review_volume": VOLUME,
    "03_temporal_volume": TEMPORAL,
    "04_customer_activity": CUSTOMER,
    "05_full": FULL,
    "06_2021_activity_only": ["reviews_first_6m_2021", "reviews_recent_6m_2021", "tips_2021"],
    "07_ratings_plus_missingness": ["business_stars", "avg_review_stars_2021", "rating_change_recent_vs_first_6m"],
    "08_business_stars_only": ["business_stars"],
}


def load_and_validate(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = {"business_id", "is_open", *[f for v in FEATURE_GROUPS.values() for f in v]}
    absent = sorted(cols.difference(df.columns))
    if absent:
        raise ValueError(f"Missing columns: {absent}")
    if df.business_id.isna().any() or df.business_id.duplicated().any():
        raise ValueError("Missing or duplicated business_id")
    if not df.is_open.isin([0, 1]).all():
        raise ValueError("is_open must contain only 0/1 values")
    for col in cols - {"business_id", "is_open"}:
        df[col] = pd.to_numeric(df[col], errors="raise")
    if not (df.reviews_2021 == df.reviews_first_6m_2021 + df.reviews_recent_6m_2021).all():
        raise ValueError("2021 review counts do not reconcile to halves")
    if not (df.reviews_recent_3m_2021 <= df.reviews_recent_6m_2021).all():
        raise ValueError("Recent-three-month counts exceed recent-six-month counts")
    for c in COUNT_FEATURES:
        if df[c].isna().any() or not np.isfinite(df[c]).all() or (df[c] % 1 != 0).any():
            raise ValueError(f"Invalid count: {c}")
        if (df[c] < 0).any():
            raise ValueError(f"Negative count: {c}")
    return df


def estimator(model_name: str, columns: list[str], trees: int, seed: int) -> Pipeline:
    """Fit imputation and transformations only inside the training folds."""
    counts = [c for c in columns if c in COUNT_FEATURES]
    numeric = [c for c in columns if c not in COUNT_FEATURES]
    if model_name == "logistic_regression":
        steps = []
        if counts:
            steps.append(("counts", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("log1p", FunctionTransformer(np.log1p)),
                ("scale", StandardScaler()),
            ]), counts))
        if numeric:
            steps.append(("numeric", Pipeline([
                ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                ("scale", StandardScaler()),
            ]), numeric))
        preprocessor = ColumnTransformer(steps)
        classifier = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
    else:
        preprocessor = ColumnTransformer([
            ("numeric", SimpleImputer(strategy="median", add_indicator=True), columns)
        ])
        classifier = RandomForestClassifier(
            n_estimators=trees, min_samples_leaf=5, max_features="sqrt",
            class_weight="balanced_subsample", n_jobs=-1, random_state=seed,
        )
    return Pipeline([("preprocess", preprocessor), ("classifier", classifier)])


def scores(y_true: pd.Series, prob: np.ndarray, pred: np.ndarray) -> dict:
    """Positive class is recorded CLOSED (is_open == 0)."""
    return {
        "n": int(len(y_true)), "closed_n": int(y_true.sum()),
        "closed_prevalence": float(y_true.mean()),
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "precision_closed": precision_score(y_true, pred, zero_division=0),
        "recall_closed": recall_score(y_true, pred, zero_division=0),
        "f1_closed": f1_score(y_true, pred, zero_division=0),
        "roc_auc_closed": roc_auc_score(y_true, prob),
        "average_precision_closed": average_precision_score(y_true, prob),
    }


def audit(df: pd.DataFrame, output: Path) -> None:
    records = []
    for status, group in df.groupby("is_open"):
        total = len(group)
        for label, mask in {
            "zero_reviews_2021": group.reviews_2021.eq(0),
            "any_reviews_2021": group.reviews_2021.gt(0),
            "zero_first_6m": group.reviews_first_6m_2021.eq(0),
            "zero_recent_6m": group.reviews_recent_6m_2021.eq(0),
            "zero_recent_3m": group.reviews_recent_3m_2021.eq(0),
            "any_tips_2021": group.tips_2021.gt(0),
            "reviews_2021_exceed_snapshot_review_count": group.reviews_2021.gt(group.business_review_count),
        }.items():
            n = int(mask.sum())
            records.append({"is_open": status, "status": "open" if status else "closed",
                            "check": label, "n": n, "denominator": total, "percent": 100 * n / total})
    pd.DataFrame(records).to_csv(output / "status_activity_audit.csv", index=False)

    # Ratings are undefined rather than zero when no reviews were observed.
    rating_checks = {
        "avg_review_stars_2021": "reviews_2021",
        "avg_review_stars_first_6m": "reviews_first_6m_2021",
        "avg_review_stars_recent_6m": "reviews_recent_6m_2021",
    }
    for rating, count in rating_checks.items():
        if rating in df.columns and not df[rating].isna().equals(df[count].eq(0)):
            raise ValueError(f"Unexpected missingness relation: {rating} vs {count}")
    print("Feature audit saved. 2021 review counts reconcile.")


# Controlled comparisons: activity uses temporal counts without redundant annual total.
ACTIVITY = [*TEMPORAL, "all_time_checkins", "tips_2021"]
RATINGS = ["avg_review_stars_2021", "rating_change_recent_vs_first_6m"]
SENTIMENT = ["sentiment_mean_2021", "sentiment_std_2021", "negative_review_share_2021",
             "sentiment_change_recent_vs_first_6m"]
EXPERIMENT_GROUPS = {
    "A_basic": BUSINESS, "B_volume": VOLUME, "C_temporal": TEMPORAL,
    "D_activity": ACTIVITY, "E_activity_ratings": ACTIVITY + RATINGS,
    "F_activity_sentiment": ACTIVITY + SENTIMENT,
    "G_full": ACTIVITY + RATINGS + SENTIMENT,
}
PAIRS = [("A_basic", "B_volume"), ("B_volume", "C_temporal"),
         ("C_temporal", "D_activity"), ("D_activity", "E_activity_ratings"),
         ("D_activity", "F_activity_sentiment"), ("E_activity_ratings", "G_full")]

def paired_intervals(y, predictions, pairs, repetitions=500):
    # Stratified paired test bootstrap: conditional on the fitted models and holdout.
    # Captures holdout sampling uncertainty, not training/model selection uncertainty.
    rng = np.random.default_rng(42)
    strata = [np.flatnonzero(y == value) for value in (0, 1)]
    draws = [np.concatenate([rng.choice(a, len(a), replace=True) for a in strata])
             for _ in range(repetitions)]
    measures = {"roc_auc_closed": roc_auc_score, "average_precision_closed": average_precision_score,
                "balanced_accuracy": lambda y, p: balanced_accuracy_score(y, p >= .5),
                "f1_closed": lambda y, p: f1_score(y, p >= .5, zero_division=0)}
    rows = []
    for model in ("logistic_regression", "random_forest"):
        for before, after in pairs:
            if (model, before) not in predictions or (model, after) not in predictions:
                continue
            a, b = predictions[model, before], predictions[model, after]
            for name, metric in measures.items():
                differences = [metric(y[i], b[i])-metric(y[i], a[i]) for i in draws]
                low, high = np.quantile(differences, [.025, .975])
                rows.append(dict(model=model, before=before, after=after, metric=name,
                                 delta=metric(y, b)-metric(y, a), ci_low=low, ci_high=high,
                                 bootstrap_repetitions=repetitions))
    return pd.DataFrame(rows)

def experiment(df, output, groups, folds=3, trees=120, bootstrap=500):
    import json
    from sklearn.metrics import confusion_matrix
    output.mkdir(parents=True, exist_ok=True)
    df = df.sort_values("business_id").reset_index(drop=True)
    y = (df.is_open == 0).astype(int)
    train, test = train_test_split(df.index, test_size=.2, stratify=y, random_state=42)
    pd.DataFrame({"business_id": df.business_id, "is_open": df.is_open,
                  "split": np.where(df.index.isin(test), "test", "train")}).to_csv(output / "split.csv", index=False)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    results, prediction_rows, probabilities = [], [], {}
    def record(model, group, prob, extra=None):
        pred = (prob >= .5).astype(int)
        row = scores(y.loc[test], prob, pred)
        tn, fp, fn, tp = confusion_matrix(y.loc[test], pred, labels=[0, 1]).ravel()
        row.update(model=model, feature_group=group, tn_open=int(tn), fp_closed=int(fp),
                   fn_closed=int(fn), tp_closed=int(tp))
        row.update(extra or {})
        results.append(row)
        prediction_rows.append(pd.DataFrame({"business_id": df.loc[test, "business_id"],
            "actual_closed": y.loc[test], "probability_closed": prob,
            "predicted_closed": pred, "model": model, "feature_group": group}))
        print(f"{output.name} {model} {group}: AUC={row['roc_auc_closed']:.4f} AP={row['average_precision_closed']:.4f}", flush=True)
    majority = float(y.loc[train].mean() >= .5)
    record("baseline", "majority", np.full(len(test), majority))
    record("baseline", "zero_review_rule", df.loc[test, "reviews_2021"].eq(0).astype(float).to_numpy())
    for group, columns in groups.items():
        for kind in ("logistic_regression", "random_forest"):
            model = estimator(kind, columns, trees, 42)
            cross = cross_validate(model, df.loc[train, columns], y.loc[train], cv=cv,
                scoring={"auc": "roc_auc", "ap": "average_precision",
                         "balanced_accuracy": "balanced_accuracy", "f1": "f1"},
                error_score="raise", n_jobs=1)
            model.fit(df.loc[train, columns], y.loc[train])
            prob = model.predict_proba(df.loc[test, columns])[:, 1]
            probabilities[kind, group] = prob
            extra = {f"cv_{metric}_{stat}": float(getattr(cross[f"test_{metric}"], stat)())
                     for metric in ("auc", "ap", "balanced_accuracy", "f1") for stat in ("mean", "std")}
            record(kind, group, prob, extra)
    result_frame = pd.DataFrame(results)
    result_frame.to_csv(output / "model_comparison.csv", index=False)
    if output.name == "main":
        result_frame.loc[result_frame.model.ne("baseline")].to_csv(output / "feature_ablation.csv", index=False)
    pd.concat(prediction_rows).to_csv(output / "predictions.csv", index=False)
    paired_intervals(y.loc[test].to_numpy(), probabilities, PAIRS, bootstrap).to_csv(output / "incremental_contributions.csv", index=False)
    (output / "metadata.json").write_text(json.dumps(dict(n=len(df), closed=int(y.sum()),
        open=int((1-y).sum()), prevalence_closed=float(y.mean()), train_n=len(train), test_n=len(test),
        seed=42, folds=folds, trees=trees, groups=groups,
        bootstrap="stratified paired holdout percentile 95%; conditional on fitted models; no multiplicity adjustment"), indent=2))

def main():
    import hashlib
    import json
    import platform
    import sklearn
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/business_features_with_sentiment.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "research/status_experiments")
    parser.add_argument("--trees", type=int, default=120)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--bootstrap", type=int, default=500)
    args = parser.parse_args()
    if args.folds < 2 or args.trees < 1 or args.bootstrap < 100:
        parser.error("Require folds >= 2, trees >= 1, bootstrap >= 100")
    df = load_and_validate(args.input)
    args.output.mkdir(parents=True, exist_ok=True)
    audit(df, args.output)
    for field in ["sentiment_mean_2021", "negative_review_share_2021"]:
        if not df[field].isna().equals(df.reviews_2021.eq(0)):
            raise ValueError("Unexpected sentiment missingness")
    if not df.sentiment_review_count_2021.eq(df.reviews_2021).all():
        raise ValueError("Sentiment counts disagree")
    experiment(df, args.output / "main", EXPERIMENT_GROUPS, args.folds, args.trees, args.bootstrap)
    active = df.loc[df.reviews_2021.gt(0)].copy()
    experiment(active, args.output / "active", {g: EXPERIMENT_GROUPS[g] for g in
        ["D_activity", "E_activity_ratings", "F_activity_sentiment", "G_full"]}, args.folds, args.trees, args.bootstrap)
    (args.output / "environment.json").write_text(json.dumps(dict(python=platform.python_version(),
        pandas=pd.__version__, numpy=np.__version__, sklearn=sklearn.__version__,
        input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest()), indent=2))

if __name__ == "__main__": main()
