
from pathlib import Path

import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# Paths and configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

INPUT_FILE = DATA_DIR / "business_features.csv"
METRICS_FILE = DATA_DIR / "baseline_model_metrics.csv"
PREDICTIONS_FILE = DATA_DIR / "baseline_test_predictions.csv"

TEST_SIZE = 0.20
RANDOM_STATE = 42

# Target: 1 = currently marked open, 0 = currently marked closed.
TARGET = "is_open"

BASIC_FEATURES = [
    "business_stars",
    "business_review_count",
]

ACTIVITY_FEATURES = BASIC_FEATURES + [
    "reviews_2021",
    "reviews_first_6m_2021",
    "reviews_recent_6m_2021",
    "reviews_recent_3m_2021",
    "avg_review_stars_2021",
    "avg_review_stars_first_6m",
    "avg_review_stars_recent_6m",
    "rating_change_recent_vs_first_6m",
    "all_time_checkins",
    "tips_2021",
    "distinct_tip_users_2021",
]


def make_model():
    """Build a preprocessing and classification pipeline."""
    return Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                add_indicator=True,
            ),
        ),
        ("scaler", StandardScaler()),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        ),
    ])


def evaluate_model(name, model, X_test, y_test):
    """Print metrics and return predictions and a metrics row."""
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    ).ravel()

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, predictions),
        "balanced_accuracy": balanced_accuracy_score(
            y_test, predictions
        ),
        "precision_closed": precision_score(
            y_test, predictions, pos_label=0, zero_division=0
        ),
        "recall_closed": recall_score(
            y_test, predictions, pos_label=0, zero_division=0
        ),
        "f1_closed": f1_score(
            y_test, predictions, pos_label=0, zero_division=0
        ),
        "precision_open": precision_score(
            y_test, predictions, pos_label=1, zero_division=0
        ),
        "recall_open": recall_score(
            y_test, predictions, pos_label=1, zero_division=0
        ),
        "f1_open": f1_score(
            y_test, predictions, pos_label=1, zero_division=0
        ),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "average_precision_open": average_precision_score(
            y_test, probabilities
        ),
        "true_closed": int(tn),
        "false_open": int(fp),
        "false_closed": int(fn),
        "true_open": int(tp),
    }

    print("\n" + "=" * 68)
    print(name)
    print("=" * 68)

    print(f"Accuracy:              {metrics['accuracy']:.4f}")
    print(
        f"Balanced accuracy:     "
        f"{metrics['balanced_accuracy']:.4f}"
    )
    print(f"ROC-AUC:               {metrics['roc_auc']:.4f}")
    print(
        f"Average precision:     "
        f"{metrics['average_precision_open']:.4f}"
    )

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Marked closed", "Marked open"],
            zero_division=0,
        )
    )

    print("Confusion matrix (rows=actual, columns=predicted):")
    print("                 Predicted closed  Predicted open")
    print(f"Actual closed       {tn:8d}        {fp:8d}")
    print(f"Actual open         {fn:8d}        {tp:8d}")

    prediction_df = pd.DataFrame({
        "actual_is_open": y_test.to_numpy(),
        "predicted_is_open": predictions,
        "predicted_open_probability": probabilities,
    }, index=y_test.index)

    prediction_df["model"] = name

    return metrics, prediction_df


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Feature file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required = {"business_id", TARGET}
    missing_required = required - set(df.columns)

    if missing_required:
        raise ValueError(
            f"Missing required columns: {sorted(missing_required)}"
        )

    if df["business_id"].duplicated().any():
        raise ValueError("Duplicate business IDs detected.")

    # Check that the required feature columns exist.
    all_features = list(dict.fromkeys(
        BASIC_FEATURES + ACTIVITY_FEATURES
    ))
    missing_features = [
        col for col in all_features if col not in df.columns
    ]

    if missing_features:
        raise ValueError(
            "Missing feature columns:\n"
            + "\n".join(missing_features)
        )

    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df[df[TARGET].isin([0, 1])].copy()
    df[TARGET] = df[TARGET].astype(int)

    for col in all_features:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Missing labels cannot be used for supervised learning.
    df = df.dropna(subset=[TARGET])

    if df[all_features].isna().all().any():
        bad_columns = df[all_features].columns[
            df[all_features].isna().all()
        ].tolist()
        raise ValueError(
            f"These features are entirely missing: {bad_columns}"
        )

    X = df[all_features]
    y = df[TARGET]

    # One common split for both models.
    train_idx, test_idx = train_test_split(
        df.index,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("=" * 68)
    print("BASELINE: CURRENT YELP BUSINESS STATUS")
    print("=" * 68)
    print(f"Businesses:       {len(df):,}")
    print(f"Training rows:    {len(train_idx):,}")
    print(f"Test rows:        {len(test_idx):,}")
    print(f"Marked closed:    {(y == 0).sum():,}")
    print(f"Marked open:      {(y == 1).sum():,}")
    print(
        "Closed share:     "
        f"{100 * (y == 0).mean():.2f}%"
    )
    print("\nModel features:")
    print(f"Basic:            {BASIC_FEATURES}")
    print(f"Activity-enhanced: {ACTIVITY_FEATURES}")

    # Majority-class baseline: always predict the most common class.
    y_test = y.loc[test_idx]
    majority_class = int(y.loc[train_idx].mode().iloc[0])
    majority_predictions = [majority_class] * len(y_test)

    print("\n" + "=" * 68)
    print("MAJORITY-CLASS BASELINE")
    print("=" * 68)
    print(
        f"Always predicts: "
        f"{'open' if majority_class == 1 else 'closed'}"
    )
    print(
        f"Accuracy: {accuracy_score(y_test, majority_predictions):.4f}"
    )
    print(
        "Balanced accuracy: "
        f"{balanced_accuracy_score(y_test, majority_predictions):.4f}"
    )

    results = []
    all_predictions = []

    for name, feature_columns in [
        ("Basic logistic regression", BASIC_FEATURES),
        ("Activity-enhanced logistic regression", ACTIVITY_FEATURES),
    ]:
        model = make_model()

        model.fit(
            df.loc[train_idx, feature_columns],
            y.loc[train_idx],
        )

        metrics, predictions = evaluate_model(
            name,
            model,
            df.loc[test_idx, feature_columns],
            y_test,
        )

        results.append(metrics)

        predictions["business_id"] = df.loc[
            predictions.index, "business_id"
        ]
        all_predictions.append(predictions.reset_index(drop=True))

    # Save outputs.
    pd.DataFrame(results).to_csv(METRICS_FILE, index=False)

    pd.concat(
        all_predictions,
        ignore_index=True,
    ).to_csv(PREDICTIONS_FILE, index=False)

    print("\n" + "=" * 68)
    print("MODEL COMPARISON")
    print("=" * 68)

    comparison = pd.DataFrame(results).set_index("model")
    print(
        comparison[
            [
                "balanced_accuracy",
                "precision_closed",
                "recall_closed",
                "f1_closed",
                "roc_auc",
                "average_precision_open",
            ]
        ].round(4).to_string()
    )

    print("\nSaved metrics to:")
    print(METRICS_FILE)
    print("\nSaved test predictions to:")
    print(PREDICTIONS_FILE)

    print("\nLIMITATIONS")
    print("- The target is current snapshot status, not future closure.")
    print("- Snapshot features may contain information accumulated")
    print("  after a hypothetical prediction date.")
    print("- Review inactivity may reflect Yelp usage, not closure.")
    print("- Performance does not establish causation or early warning.")
    print("- The test split is random, not a temporal holdout.")


if __name__ == "__main__":
    main()
