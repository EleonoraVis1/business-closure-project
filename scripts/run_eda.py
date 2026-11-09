
"""
Exploratory Data Analysis for Yelp Restaurant Operating Status.

Run from the project root:
    python -m scripts.run_eda

Input:
    data/business_features_stage1_final.csv

Output:
    data/eda/
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import ttest_ind, mannwhitneyu


# ============================================================
# 1. PATHS AND CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "business_features_stage1_final.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "eda"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# Features organized by research category.
FEATURE_GROUPS = {
    "Review activity": [
        "business_review_count",
        "reviews_2021",
        "reviews_first_6m_2021",
        "reviews_recent_6m_2021",
        "reviews_recent_3m_2021",
    ],

    "Ratings": [
        "business_stars",
        "avg_review_stars_2021",
        "avg_review_stars_first_6m",
        "avg_review_stars_recent_6m",
        "rating_change_recent_vs_first_6m",
    ],

    "Customer activity": [
        "all_time_checkins",
        "tips_2021",
        "distinct_tip_users_2021",
    ],
}

COUNT_FEATURES = [
    "business_review_count",
    "reviews_2021",
    "reviews_first_6m_2021",
    "reviews_recent_6m_2021",
    "reviews_recent_3m_2021",
    "all_time_checkins",
    "tips_2021",
    "distinct_tip_users_2021",
]

RATING_FEATURES = [
    "business_stars",
    "avg_review_stars_2021",
    "avg_review_stars_first_6m",
    "avg_review_stars_recent_6m",
    "rating_change_recent_vs_first_6m",
]


# ============================================================
# 2. LOAD AND VALIDATE DATA
# ============================================================

print("=" * 65)
print("YELP RESTAURANT EDA")
print("=" * 65)

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

print(f"\nLoaded dataset: {df.shape[0]:,} rows, {df.shape[1]} columns")

if "business_id" not in df.columns:
    raise ValueError("business_id column is missing.")

if "is_open" not in df.columns:
    raise ValueError("is_open column is missing.")

if df["business_id"].duplicated().any():
    raise ValueError("Duplicate business IDs detected.")

if df["is_open"].isna().any():
    raise ValueError("Missing target labels detected.")

if not set(df["is_open"].unique()).issubset({0, 1}):
    raise ValueError("Unexpected values in is_open.")

# Retain only features present in the dataset.
FEATURE_GROUPS = {
    group: [f for f in features if f in df.columns]
    for group, features in FEATURE_GROUPS.items()
}

FEATURES = [
    feature
    for features in FEATURE_GROUPS.values()
    for feature in features
]

missing_columns = [
    feature
    for features in FEATURE_GROUPS.values()
    for feature in features
    if feature not in df.columns
]

if missing_columns:
    print("Warning: Some expected features are missing.")

print("\nTarget distribution:")
print(df["is_open"].value_counts().sort_index())

open_count = int((df["is_open"] == 1).sum())
closed_count = int((df["is_open"] == 0).sum())

print(f"\nOpen: {open_count:,} ({open_count / len(df) * 100:.2f}%)")
print(f"Closed: {closed_count:,} ({closed_count / len(df) * 100:.2f}%)")


# ============================================================
# 3. MISSING VALUE ANALYSIS
# ============================================================

print("\n" + "=" * 65)
print("MISSING VALUES")
print("=" * 65)

missing = pd.DataFrame({
    "feature": df.columns,
    "missing_count": df.isna().sum().values,
    "missing_percentage": (df.isna().mean() * 100).values,
})

missing = missing.sort_values(
    "missing_percentage",
    ascending=False
)

missing.to_csv(
    OUTPUT_DIR / "eda_missing_values.csv",
    index=False
)

print(missing.to_string(index=False))


# ============================================================
# 4. DESCRIPTIVE STATISTICS BY STATUS
# ============================================================

print("\n" + "=" * 65)
print("DESCRIPTIVE STATISTICS")
print("=" * 65)

summary_rows = []

for feature in FEATURES:

    for status, label in [(1, "Open"), (0, "Closed")]:

        values = df.loc[df["is_open"] == status, feature]

        summary_rows.append({
            "feature": feature,
            "status": label,
            "business_count": len(values),
            "nonmissing_count": values.notna().sum(),
            "missing_count": values.isna().sum(),
            "missing_percentage": values.isna().mean() * 100,
            "mean": values.mean(),
            "std": values.std(),
            "min": values.min(),
            "q1": values.quantile(0.25),
            "median": values.median(),
            "q3": values.quantile(0.75),
            "max": values.max(),
        })

summary = pd.DataFrame(summary_rows)

summary.to_csv(
    OUTPUT_DIR / "eda_summary_by_status.csv",
    index=False
)

print("Saved descriptive statistics.")


# ============================================================
# 5. OPEN VS CLOSED STATISTICAL COMPARISON
# ============================================================

print("\n" + "=" * 65)
print("OPEN VS CLOSED COMPARISON")
print("=" * 65)


def calculate_cohens_d(open_values, closed_values):
    """Calculate pooled-standard-deviation Cohen's d."""

    n1 = len(open_values)
    n0 = len(closed_values)

    if n1 < 2 or n0 < 2:
        return np.nan

    var1 = open_values.var(ddof=1)
    var0 = closed_values.var(ddof=1)

    pooled_variance = (
        ((n1 - 1) * var1 + (n0 - 1) * var0)
        / (n1 + n0 - 2)
    )

    if pooled_variance <= 0:
        return np.nan

    pooled_sd = np.sqrt(pooled_variance)

    return (open_values.mean() - closed_values.mean()) / pooled_sd


comparison_rows = []

for feature in FEATURES:

    open_values = (
        df.loc[df["is_open"] == 1, feature]
        .dropna()
        .astype(float)
    )

    closed_values = (
        df.loc[df["is_open"] == 0, feature]
        .dropna()
        .astype(float)
    )

    if len(open_values) < 2 or len(closed_values) < 2:
        continue

    # Welch's t-test does not assume equal group variances.
    welch = ttest_ind(
        open_values,
        closed_values,
        equal_var=False
    )

    # Mann-Whitney U compares the distributions/ranks.
    mw = mannwhitneyu(
        open_values,
        closed_values,
        alternative="two-sided"
    )

    # Probability that a randomly selected open value exceeds
    # a randomly selected closed value, with ties half-weighted.
    probability = mw.statistic / (
        len(open_values) * len(closed_values)
    )

    comparison_rows.append({
        "feature": feature,

        "open_n": len(open_values),
        "closed_n": len(closed_values),

        "open_mean": open_values.mean(),
        "closed_mean": closed_values.mean(),

        "mean_difference": (
            open_values.mean() - closed_values.mean()
        ),

        "open_median": open_values.median(),
        "closed_median": closed_values.median(),

        "median_difference": (
            open_values.median() - closed_values.median()
        ),

        "cohens_d": calculate_cohens_d(
            open_values,
            closed_values
        ),

        "welch_p_value": welch.pvalue,
        "mann_whitney_p_value": mw.pvalue,

        "probability_open_exceeds_closed": probability,

        "rank_biserial_correlation": 2 * probability - 1,
    })


comparisons = pd.DataFrame(comparison_rows)


# Benjamini-Hochberg correction for multiple Welch tests.
def benjamini_hochberg(p_values):

    p_values = np.asarray(p_values, dtype=float)

    adjusted = np.full(len(p_values), np.nan)

    valid = np.isfinite(p_values)

    if not valid.any():
        return adjusted

    p = p_values[valid]

    order = np.argsort(p)
    sorted_p = p[order]

    m = len(sorted_p)

    ranked_adjusted = (
        sorted_p * m / np.arange(1, m + 1)
    )

    ranked_adjusted = np.minimum.accumulate(
        ranked_adjusted[::-1]
    )[::-1]

    ranked_adjusted = np.minimum(ranked_adjusted, 1.0)

    restored = np.empty(m)
    restored[order] = ranked_adjusted

    adjusted[valid] = restored

    return adjusted


comparisons["welch_fdr_q_value"] = benjamini_hochberg(
    comparisons["welch_p_value"]
)

comparisons.to_csv(
    OUTPUT_DIR / "eda_feature_comparisons.csv",
    index=False
)

print(
    comparisons.sort_values(
        "cohens_d",
        key=lambda x: x.abs(),
        ascending=False
    ).to_string(index=False)
)


# ============================================================
# 6. CORRELATION ANALYSIS
# ============================================================

print("\n" + "=" * 65)
print("CORRELATION ANALYSIS")
print("=" * 65)

# Spearman correlation is useful for skewed count variables
# because it measures monotonic association using ranks.
correlation = df[FEATURES].corr(
    method="spearman",
    min_periods=100
)

correlation.to_csv(
    OUTPUT_DIR / "eda_correlation_matrix.csv"
)

status_correlations = (
    df[["is_open"] + FEATURES]
    .corr(method="spearman")["is_open"]
    .drop("is_open")
    .sort_values(key=lambda x: x.abs(), ascending=False)
)

status_correlations.to_csv(
    OUTPUT_DIR / "eda_status_correlations.csv"
)

print("\nFeature correlations with recorded status:")
print(status_correlations.to_string())

# Extract strongest unique feature pairs.
upper_triangle = np.triu(
    np.ones(correlation.shape),
    k=1
).astype(bool)

correlation_pairs = (
    correlation.where(upper_triangle)
    .stack()
    .rename("spearman_correlation")
    .reset_index()
)

correlation_pairs.columns = [
    "feature_1",
    "feature_2",
    "spearman_correlation"
]

correlation_pairs["absolute_correlation"] = (
    correlation_pairs["spearman_correlation"].abs()
)

correlation_pairs = correlation_pairs.sort_values(
    "absolute_correlation",
    ascending=False
)

correlation_pairs.to_csv(
    OUTPUT_DIR / "eda_strongest_correlations.csv",
    index=False
)

print("\nStrongest feature relationships:")
print(correlation_pairs.head(15).to_string(index=False))


# ============================================================
# 7. DISTRIBUTION PLOTS
# ============================================================

print("\nGenerating distribution plots...")

for feature in FEATURES:

    # Count variables are highly skewed, so use log1p.
    use_log = feature in COUNT_FEATURES

    fig, ax = plt.subplots(figsize=(9, 5))

    for status, label in [(1, "Open"), (0, "Closed")]:

        values = df.loc[
            df["is_open"] == status,
            feature
        ].dropna()

        if use_log:
            values = np.log1p(values)

        ax.hist(
            values,
            bins=40,
            alpha=0.45,
            density=True,
            label=label
        )

    ax.set_title(
        f"Distribution of {feature} by restaurant status"
    )

    if use_log:
        ax.set_xlabel(f"log(1 + {feature})")
    else:
        ax.set_xlabel(feature)

    ax.set_ylabel("Density")
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / f"distribution_{feature}.png",
        dpi=150
    )

    plt.close(fig)


# ============================================================
# 8. BOXPLOTS
# ============================================================

print("Generating boxplots...")

for feature in FEATURES:

    use_log = feature in COUNT_FEATURES

    open_values = df.loc[
        df["is_open"] == 1,
        feature
    ].dropna()

    closed_values = df.loc[
        df["is_open"] == 0,
        feature
    ].dropna()

    if use_log:
        open_values = np.log1p(open_values)
        closed_values = np.log1p(closed_values)

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.boxplot(
        [open_values, closed_values],
        tick_labels=["Open", "Closed"],
        showfliers=False
    )

    ax.set_title(
        f"{feature}: open vs closed"
    )

    if use_log:
        ax.set_ylabel(f"log(1 + {feature})")
    else:
        ax.set_ylabel(feature)

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / f"boxplot_{feature}.png",
        dpi=150
    )

    plt.close(fig)


# ============================================================
# 9. CORRELATION HEATMAP
# ============================================================

print("Generating correlation heatmap...")

fig, ax = plt.subplots(figsize=(13, 10))

image = ax.imshow(
    correlation,
    vmin=-1,
    vmax=1,
    aspect="auto"
)

ax.set_xticks(range(len(FEATURES)))
ax.set_yticks(range(len(FEATURES)))

ax.set_xticklabels(
    FEATURES,
    rotation=90
)

ax.set_yticklabels(FEATURES)

ax.set_title("Spearman Correlation Matrix")

fig.colorbar(
    image,
    ax=ax,
    label="Spearman correlation"
)

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "correlation_heatmap.png",
    dpi=160
)

plt.close(fig)


# ============================================================
# 10. AUTOMATIC FINDINGS REPORT
# ============================================================

print("Creating findings report...")

report = []

report.append("# Yelp Restaurant Operating Status: EDA Findings")
report.append("")

report.append("## 1. Dataset overview")
report.append("")

report.append(f"- Total restaurants: {len(df):,}")
report.append(f"- Open: {open_count:,} ({open_count / len(df) * 100:.2f}%)")
report.append(f"- Closed: {closed_count:,} ({closed_count / len(df) * 100:.2f}%)")
report.append("")

report.append("## 2. Largest open/closed differences")
report.append("")

report.append(
    "Features are ordered by the absolute value of Cohen's d. "
    "Positive values mean the open group has a higher mean."
)

report.append("")

top_effects = comparisons.assign(
    absolute_d=comparisons["cohens_d"].abs()
).sort_values(
    "absolute_d",
    ascending=False
)

for _, row in top_effects.head(10).iterrows():

    report.append(
        f"- **{row['feature']}**: "
        f"open mean = {row['open_mean']:.3f}, "
        f"closed mean = {row['closed_mean']:.3f}; "
        f"open median = {row['open_median']:.3f}, "
        f"closed median = {row['closed_median']:.3f}; "
        f"Cohen's d = {row['cohens_d']:.3f}; "
        f"Welch FDR q = {row['welch_fdr_q_value']:.4g}."
    )

report.append("")
report.append("## 3. Strongest feature relationships")
report.append("")

for _, row in correlation_pairs.head(10).iterrows():

    report.append(
        f"- {row['feature_1']} and {row['feature_2']}: "
        f"Spearman rho = {row['spearman_correlation']:.3f}."
    )

report.append("")
report.append("## 4. Interpretation and limitations")
report.append("")

report.append(
    "- The target is Yelp's recorded operating status, not a verified "
    "closure date or future closure event."
)

report.append(
    "- These are observational comparisons and do not establish causation."
)

report.append(
    "- Count features are highly likely to be skewed; log1p plots are "
    "included to make their distributions easier to inspect."
)

report.append(
    "- Missing review averages are not zero-star ratings. "
    "They should remain missing during descriptive analysis."
)

report.append(
    "- All-time check-ins have a different time window from 2021 reviews "
    "and tips."
)

report.append(
    "- Statistical significance should not be confused with practical "
    "importance. Examine effect sizes and distributions."
)

report.append(
    "- Any imputation, scaling, encoding, or feature selection for "
    "classification must be fitted using training data only."
)

report.append("")
report.append("## 5. Questions for the next stage")
report.append("")

report.append(
    "1. Which features show the largest and most consistent differences "
    "between open and closed restaurants?"
)

report.append(
    "2. Are review counts, check-ins, and tips strongly redundant?"
)

report.append(
    "3. Do ratings provide information beyond customer activity?"
)

report.append(
    "4. How should missing rating averages be handled in a model pipeline?"
)

report.append(
    "5. Which feature groups should be compared in the classification "
    "experiments?"
)

report_path = OUTPUT_DIR / "eda_findings.md"

report_path.write_text(
    "\n".join(report),
    encoding="utf-8"
)


# ============================================================
# 11. FINISHED
# ============================================================

print("\n" + "=" * 65)
print("EDA COMPLETED")
print("=" * 65)

print(f"\nOutput directory:\n{OUTPUT_DIR}")

print("\nGenerated files:")
for file in sorted(OUTPUT_DIR.iterdir()):
    print(f"  {file.name}")

print("\nNext step: inspect the comparisons, distributions, and correlations.")
