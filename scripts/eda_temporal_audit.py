
from pathlib import Path

import pandas as pd


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ANALYSIS_DIR = PROJECT_ROOT / "analysis"

TIMELINE_FILE = ANALYSIS_DIR / "eda_business_timeline.csv"
FEATURES_FILE = DATA_DIR / "business_features.csv"

FEATURE_START = pd.Timestamp("2021-01-01")
FEATURE_END = pd.Timestamp("2021-12-31")
RECENT_START = pd.Timestamp("2021-10-01")


def require_columns(df, columns, filename):
    """Raise a clear error if expected columns are missing."""
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(
            f"{filename} is missing expected columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )


def find_column(df, candidates, filename):
    """Find the first available column from a list of alternatives."""
    for candidate in candidates:
        if candidate in df.columns:
            return candidate

    raise ValueError(
        f"Could not find any of these columns in {filename}: {candidates}\n"
        f"Available columns: {list(df.columns)}"
    )


def pct(numerator, denominator):
    """Percentage with protection against division by zero."""
    if denominator == 0:
        return float("nan")
    return 100.0 * numerator / denominator


def print_section(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# ============================================================
# Load files
# ============================================================

def main():
    if not TIMELINE_FILE.exists():
        raise FileNotFoundError(
            f"Timeline file not found: {TIMELINE_FILE}"
        )

    if not FEATURES_FILE.exists():
        raise FileNotFoundError(
            f"Feature file not found: {FEATURES_FILE}"
        )

    timeline = pd.read_csv(TIMELINE_FILE)
    features = pd.read_csv(FEATURES_FILE)

    require_columns(
        features,
        ["business_id", "is_open", "reviews_2021",
         "reviews_recent_3m_2021"],
        FEATURES_FILE.name,
    )

    # The timeline script may use different names for the same concept.
    business_id_col = find_column(
        timeline,
        ["business_id"],
        TIMELINE_FILE.name,
    )

    last_date_col = find_column(
        timeline,
        [
            "last_review_date",
            "last_review",
            "last_review_month",
            "last_date",
            "max_review_date",
        ],
        TIMELINE_FILE.name,
    )

    timeline[last_date_col] = pd.to_datetime(
        timeline[last_date_col],
        errors="coerce",
    )

    # Keep one timeline row per business. If there are duplicate rows,
    # use the latest observed date rather than arbitrarily selecting one.
    timeline = (
        timeline.groupby(business_id_col, as_index=False)
        [last_date_col]
        .max()
        .rename(
            columns={
                business_id_col: "business_id",
                last_date_col: "last_review_date",
            }
        )
    )

    features["is_open"] = pd.to_numeric(
        features["is_open"], errors="coerce"
    )
    features["reviews_2021"] = pd.to_numeric(
        features["reviews_2021"], errors="coerce"
    )
    features["reviews_recent_3m_2021"] = pd.to_numeric(
        features["reviews_recent_3m_2021"], errors="coerce"
    )

    # Merge the two existing outputs.
    merged = features.merge(
        timeline,
        on="business_id",
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    print_section("TEMPORAL AUDIT: INPUT VALIDATION")

    print(f"Feature rows:                 {len(features):,}")
    print(f"Timeline businesses:          {len(timeline):,}")
    print(f"Merged rows:                  {len(merged):,}")
    print(
        "Feature IDs missing timeline: "
        f"{(merged['_merge'] == 'left_only').sum():,}"
    )
    print(
        "Timeline IDs missing features: "
        f"{len(set(timeline['business_id']) - set(features['business_id'])):,}"
    )
    print(
        "Missing last-review dates:    "
        f"{merged['last_review_date'].isna().sum():,}"
    )
    print(
        "Duplicate feature IDs:        "
        f"{features['business_id'].duplicated().sum():,}"
    )

    # Current status label: 1 = marked open, 0 = marked closed.
    status_names = {0: "Marked closed", 1: "Marked open"}

    print_section("1. LAST OBSERVED REVIEW DATE BY STATUS")

    for status, name in status_names.items():
        group = merged[merged["is_open"] == status]
        valid_dates = group["last_review_date"].dropna()

        print(f"\n{name} (n={len(group):,})")

        if valid_dates.empty:
            print("  No valid last-review dates.")
            continue

        print(f"  Missing last-review date: {group['last_review_date'].isna().sum():,}")
        print(f"  Earliest last review:      {valid_dates.min().date()}")
        print(f"  Median last review:        {valid_dates.median().date()}")
        print(f"  Latest last review:        {valid_dates.max().date()}")

        # Distribution by last observed review date.
        categories = {
            "Before 2021": valid_dates < FEATURE_START,
            "During 2021": (
                (valid_dates >= FEATURE_START)
                & (valid_dates <= FEATURE_END)
            ),
            "After 2021": valid_dates > FEATURE_END,
        }

        print("  Last-review date distribution:")
        for label, mask in categories.items():
            count = int(mask.sum())
            print(
                f"    {label:<15} {count:>7,} "
                f"({pct(count, len(valid_dates)):5.1f}%)"
            )

    print_section("2. BUSINESSES WITH REVIEWS DURING 2021")

    for status, name in status_names.items():
        group = merged[merged["is_open"] == status]
        n = len(group)

        has_2021_reviews = group["reviews_2021"].fillna(0) > 0
        has_recent_reviews = group["reviews_recent_3m_2021"].fillna(0) > 0

        print(f"\n{name} (n={n:,})")

        count_2021 = int(has_2021_reviews.sum())
        count_recent = int(has_recent_reviews.sum())

        print(
            f"  Any reviews in 2021:       {count_2021:,} "
            f"({pct(count_2021, n):.1f}%)"
        )
        print(
            f"  Any reviews Oct-Dec 2021:  {count_recent:,} "
            f"({pct(count_recent, n):.1f}%)"
        )

        # Compare review activity with the last observed review date.
        dates = group["last_review_date"]
        last_date_during_2021 = (
            dates.notna()
            & (dates >= FEATURE_START)
            & (dates <= FEATURE_END)
        )

        print(
            "  Last review during 2021:   "
            f"{int(last_date_during_2021.sum()):,} "
            f"({pct(int(last_date_during_2021.sum()), n):.1f}%)"
        )

    print_section("3. TEMPORAL RELATIONSHIP TO THE 2021 FEATURE WINDOW")

    merged["has_reviews_2021"] = merged["reviews_2021"].fillna(0) > 0
    merged["last_review_before_2021"] = (
        merged["last_review_date"] < FEATURE_START
    )
    merged["last_review_during_2021"] = (
        (merged["last_review_date"] >= FEATURE_START)
        & (merged["last_review_date"] <= FEATURE_END)
    )
    merged["last_review_after_2021"] = (
        merged["last_review_date"] > FEATURE_END
    )

    # This indicates a temporal ordering only. It does not mean the
    # business closed on that date.
    merged["last_review_at_or_before_feature_end"] = (
        merged["last_review_date"] <= FEATURE_END
    )

    print(
        "\nLast review before 2021 means no later review appears in the "
        "available review data."
    )
    print(
        "Last review during 2021 means the final observed review falls "
        "inside the feature window."
    )
    print(
        "Last review after 2021 means review activity continued after "
        "the proposed feature window."
    )

    for status, name in status_names.items():
        group = merged[merged["is_open"] == status]
        n = len(group)

        print(f"\n{name} (n={n:,})")

        for column, label in [
            ("last_review_before_2021", "Last review before 2021"),
            ("last_review_during_2021", "Last review during 2021"),
            ("last_review_after_2021", "Last review after 2021"),
            ("has_reviews_2021", "Has reviews during 2021"),
        ]:
            count = int(group[column].fillna(False).sum())
            print(
                f"  {label:<28} {count:>7,} "
                f"({pct(count, n):5.1f}%)"
            )

    # Key check: among businesses with no reviews in 2021, how often
    # does the last observed review predate the feature window?
    print_section("4. KEY CHECK: NO 2021 REVIEWS, LAST REVIEW BEFORE 2021")

    for status, name in status_names.items():
        group = merged[merged["is_open"] == status]
        no_2021_reviews = ~group["has_reviews_2021"]
        before_2021 = group["last_review_before_2021"].fillna(False)

        selected = no_2021_reviews & before_2021
        denominator = int(no_2021_reviews.sum())
        count = int(selected.sum())

        print(f"\n{name}")
        print(f"  No reviews in 2021:        {denominator:,}")
        print(
            f"  Also last review pre-2021: {count:,} "
            f"({pct(count, denominator):.1f}% of businesses with no 2021 reviews)"
        )

    print_section("5. INTERPRETATION AND MODELING DECISION")

    closed = merged[merged["is_open"] == 0]
    open_ = merged[merged["is_open"] == 1]

    closed_before = pct(
        int(closed["last_review_before_2021"].fillna(False).sum()),
        len(closed),
    )
    open_before = pct(
        int(open_["last_review_before_2021"].fillna(False).sum()),
        len(open_),
    )

    print(f"Closed businesses whose last review predates 2021: {closed_before:.1f}%")
    print(f"Open businesses whose last review predates 2021:   {open_before:.1f}%")

    print("\nAudit conclusion:")
    print(
        "- The last review date is an activity timestamp, NOT a verified "
        "closure date."
    )
    print(
        "- The current is_open label does not identify when a business "
        "closed."
    )
    print(
        "- Businesses whose last review predates 2021 may have become "
        "inactive before the feature window."
    )
    print(
        "- Businesses with reviews after 2021 show that review activity "
        "continued beyond the feature window, but this alone does not "
        "prove that they remained open."
    )
    print(
        "- Therefore, the current data supports exploratory analysis or "
        "current-status classification, not validated forward prediction "
        "of closure within a defined time horizon."
    )

    # Save an audit dataset without modifying either input.
    output_file = DATA_DIR / "eda_temporal_audit.csv"
    merged.drop(columns=["_merge"]).to_csv(output_file, index=False)

    print(f"\nSaved merged audit table to:\n{output_file}")
    print("\nAudit complete. Original files were not modified.")


if __name__ == "__main__":
    main()
