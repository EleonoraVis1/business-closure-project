
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_DIR = Path(
    r"C:\Users\zarni\Desktop\School\mining\business-closure-project"
)
DATA_DIR = PROJECT_DIR / "data"

BUSINESSES_FILE = DATA_DIR / "businesses.jsonl.gz"
REVIEWS_FILE = DATA_DIR / "reviews.jsonl.gz"
CHECKINS_FILE = DATA_DIR / "checkins.jsonl.gz"
TIPS_FILE = DATA_DIR / "tips.jsonl.gz"

OUTPUT_FILE = DATA_DIR / "business_features.csv"

# Fixed observation window: complete calendar year 2021.
WINDOW_START = pd.Period("2021-01", freq="M")
RECENT_START = pd.Period("2021-07", freq="M")
WINDOW_END = pd.Period("2021-12", freq="M")


def read_jsonl_gz(path):
    """Stream records from a compressed JSON Lines file."""
    if not path.exists():
        print(f"Optional or required file not found: {path}")
        return

    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def safe_mean(total, count):
    return total / count if count else None


# --------------------------------------------------
# 1. Load business-level information
# --------------------------------------------------

businesses = {}

for business in read_jsonl_gz(BUSINESSES_FILE):
    business_id = business.get("business_id")
    if not business_id:
        continue

    businesses[business_id] = {
        "is_open": business.get("is_open"),
        "business_stars": business.get("stars"),
        "business_review_count": business.get("review_count"),
        "city": business.get("city"),
        "state": business.get("state"),
    }

print(f"Loaded {len(businesses):,} businesses.")


# --------------------------------------------------
# 2. Aggregate review features by business and month
# --------------------------------------------------

monthly = defaultdict(
    lambda: {
        "review_count": 0,
        "rating_sum": 0.0,
        "rating_count": 0,
    }
)

reviews_processed = 0

for review in read_jsonl_gz(REVIEWS_FILE):
    business_id = review.get("business_id")
    raw_date = review.get("date")

    if business_id not in businesses or not raw_date:
        continue

    try:
        month = pd.Period(str(raw_date)[:7], freq="M")
    except (ValueError, TypeError):
        continue

    # Only collect the fixed 2021 observation window.
    if not (WINDOW_START <= month <= WINDOW_END):
        continue

    entry = monthly[(business_id, month)]
    entry["review_count"] += 1

    stars = review.get("stars")
    if isinstance(stars, (int, float)) and 1 <= stars <= 5:
        entry["rating_sum"] += stars
        entry["rating_count"] += 1

    reviews_processed += 1

    if reviews_processed % 500_000 == 0:
        print(f"Collected {reviews_processed:,} reviews in 2021...")


# --------------------------------------------------
# 3. Create business-level review features
# --------------------------------------------------

features = {}

for business_id, business in businesses.items():
    # Initialize all six months in each half of 2021.
    counts = {}
    rating_sums = {}
    rating_counts = {}

    for month in pd.period_range(WINDOW_START, WINDOW_END, freq="M"):
        entry = monthly.get((business_id, month), {})

        counts[month] = entry.get("review_count", 0)
        rating_sums[month] = entry.get("rating_sum", 0.0)
        rating_counts[month] = entry.get("rating_count", 0)

    first_half = pd.period_range(
        WINDOW_START, RECENT_START - 1, freq="M"
    )
    second_half = pd.period_range(
        RECENT_START, WINDOW_END, freq="M"
    )

    def count_in(months):
        return sum(counts[m] for m in months)

    def rating_in(months):
        total = sum(rating_sums[m] for m in months)
        n = sum(rating_counts[m] for m in months)
        return safe_mean(total, n)

    first_half_rating = rating_in(first_half)
    second_half_rating = rating_in(second_half)

    features[business_id] = {
        **business,
        "reviews_2021": count_in(
            pd.period_range(WINDOW_START, WINDOW_END, freq="M")
        ),
        "reviews_first_6m_2021": count_in(first_half),
        "reviews_recent_6m_2021": count_in(second_half),
        "reviews_recent_3m_2021": count_in(
            pd.period_range("2021-10", "2021-12", freq="M")
        ),
        "avg_review_stars_2021": rating_in(
            pd.period_range(WINDOW_START, WINDOW_END, freq="M")
        ),
        "avg_review_stars_first_6m": first_half_rating,
        "avg_review_stars_recent_6m": second_half_rating,
        "rating_change_recent_vs_first_6m": (
            second_half_rating - first_half_rating
            if first_half_rating is not None
            and second_half_rating is not None
            else None
        ),
    }


# --------------------------------------------------
# 4. Add all-time check-in counts, if available
# --------------------------------------------------
# Yelp check-in records generally contain a comma-separated
# list of timestamps in the "date" field. This is an all-time
# count, NOT a 2021-only feature.

checkin_counts = defaultdict(int)

if CHECKINS_FILE.exists():
    for record in read_jsonl_gz(CHECKINS_FILE):
        business_id = record.get("business_id")
        raw_dates = record.get("date", "")

        if business_id not in businesses or not raw_dates:
            continue

        checkin_counts[business_id] = sum(
            1 for item in str(raw_dates).split(",") if item.strip()
        )

for business_id in businesses:
    features[business_id]["all_time_checkins"] = (
        checkin_counts.get(business_id, 0)
    )


# --------------------------------------------------
# 5. Add tip counts during 2021
# --------------------------------------------------

tip_counts_2021 = defaultdict(int)
tip_users_2021 = defaultdict(set)

if TIPS_FILE.exists():
    for tip in read_jsonl_gz(TIPS_FILE):
        business_id = tip.get("business_id")
        raw_date = tip.get("date")

        if business_id not in businesses or not raw_date:
            continue

        try:
            month = pd.Period(str(raw_date)[:7], freq="M")
        except (ValueError, TypeError):
            continue

        if WINDOW_START <= month <= WINDOW_END:
            tip_counts_2021[business_id] += 1

            user_id = tip.get("user_id")
            if user_id:
                tip_users_2021[business_id].add(user_id)

for business_id in businesses:
    features[business_id]["tips_2021"] = (
        tip_counts_2021.get(business_id, 0)
    )
    features[business_id]["distinct_tip_users_2021"] = (
        len(tip_users_2021.get(business_id, set()))
    )


# --------------------------------------------------
# 6. Save the feature table
# --------------------------------------------------

columns = [
    "business_id",
    "is_open",
    "city",
    "state",
    "business_stars",
    "business_review_count",
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

with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=columns)
    writer.writeheader()

    for business_id, row in features.items():
        writer.writerow({
            "business_id": business_id,
            **{col: row.get(col) for col in columns if col != "business_id"},
        })


# --------------------------------------------------
# 7. Print validation and a brief summary
# --------------------------------------------------

df = pd.read_csv(OUTPUT_FILE)

print("\n" + "=" * 55)
print("BUSINESS FEATURE TABLE")
print("=" * 55)
print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")
print(f"Unique business IDs: {df['business_id'].nunique():,}")
print(f"Duplicate IDs: {df['business_id'].duplicated().sum():,}")
print(f"Missing labels: {df['is_open'].isna().sum():,}")

print("\nCurrent status counts:")
print(df["is_open"].value_counts(dropna=False).sort_index())

print("\nFeature summary by current status:")
summary_columns = [
    "business_stars",
    "business_review_count",
    "reviews_2021",
    "reviews_recent_3m_2021",
    "avg_review_stars_2021",
    "rating_change_recent_vs_first_6m",
    "all_time_checkins",
    "tips_2021",
]

print(
    df.groupby("is_open")[summary_columns]
    .mean()
    .round(3)
    .to_string()
)

print("\nMissing values in candidate features:")
print(df[summary_columns].isna().sum().to_string())

print(f"\nSaved feature table to:\n{OUTPUT_FILE}")
