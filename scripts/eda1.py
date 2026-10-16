import csv
import gzip
import json
from collections import defaultdict
from datetime import date
from pathlib import Path

DATA_DIR = Path(r"C:\Users\zarni\Desktop\School\mining\business-closure-project\data")

BUSINESSES_FILE = DATA_DIR / "businesses.jsonl.gz"
REVIEWS_FILE = DATA_DIR / "reviews.jsonl.gz"

MONTHLY_OUTPUT = DATA_DIR / "eda_monthly_reviews.csv"
BUSINESS_OUTPUT = DATA_DIR / "eda_business_timeline.csv"


def read_jsonl_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                yield json.loads(line)


# --------------------------------------------------
# 1. Load restaurant IDs and their current status
# --------------------------------------------------

business_status = {}

for business in read_jsonl_gz(BUSINESSES_FILE):
    business_id = business.get("business_id")
    status = business.get("is_open")

    if business_id is not None:
        business_status[business_id] = status

print(f"Restaurants loaded: {len(business_status):,}")
print("Scanning reviews. This may take a few minutes...")


# --------------------------------------------------
# 2. Initialize aggregates
# --------------------------------------------------

business_stats = {
    business_id: {
        "review_count": 0,
        "first_review": None,
        "last_review": None,
        "rating_sum": 0.0,
        "rating_count": 0,
        "undated_reviews": 0,
    }
    for business_id in business_status
}

monthly = defaultdict(
    lambda: {
        "reviews": 0,
        "rating_sum": 0.0,
        "rating_count": 0,
        "open_reviews": 0,
        "closed_reviews": 0,
        "open_rating_sum": 0.0,
        "closed_rating_sum": 0.0,
        "open_rating_count": 0,
        "closed_rating_count": 0,
        "businesses_reviewed": set(),
    }
)

total_reviews = 0
missing_business_id = 0
unknown_business_id = 0
missing_date = 0
invalid_date = 0
missing_stars = 0
earliest_date = None
latest_date = None


# --------------------------------------------------
# 3. Stream through the review dataset
# --------------------------------------------------

for review in read_jsonl_gz(REVIEWS_FILE):
    total_reviews += 1

    business_id = review.get("business_id")

    if not business_id:
        missing_business_id += 1
        continue

    if business_id not in business_stats:
        unknown_business_id += 1
        continue

    stats = business_stats[business_id]
    stats["review_count"] += 1

    stars = review.get("stars")
    if isinstance(stars, (int, float)) and 1 <= stars <= 5:
        stats["rating_sum"] += stars
        stats["rating_count"] += 1
    else:
        missing_stars += 1

    raw_date = review.get("date")

    if not raw_date:
        missing_date += 1
        stats["undated_reviews"] += 1
        continue

    try:
        review_date = date.fromisoformat(str(raw_date)[:10])
    except (ValueError, TypeError):
        invalid_date += 1
        stats["undated_reviews"] += 1
        continue

    if earliest_date is None or review_date < earliest_date:
        earliest_date = review_date

    if latest_date is None or review_date > latest_date:
        latest_date = review_date

    if (
        stats["first_review"] is None
        or review_date < stats["first_review"]
    ):
        stats["first_review"] = review_date

    if (
        stats["last_review"] is None
        or review_date > stats["last_review"]
    ):
        stats["last_review"] = review_date

    month = review_date.strftime("%Y-%m")
    entry = monthly[month]
    status = business_status[business_id]

    entry["reviews"] += 1
    entry["businesses_reviewed"].add(business_id)

    if isinstance(stars, (int, float)) and 1 <= stars <= 5:
        entry["rating_sum"] += stars
        entry["rating_count"] += 1

    if status == 1:
        entry["open_reviews"] += 1
        if isinstance(stars, (int, float)) and 1 <= stars <= 5:
            entry["open_rating_sum"] += stars
            entry["open_rating_count"] += 1

    elif status == 0:
        entry["closed_reviews"] += 1
        if isinstance(stars, (int, float)) and 1 <= stars <= 5:
            entry["closed_rating_sum"] += stars
            entry["closed_rating_count"] += 1

    if total_reviews % 500_000 == 0:
        print(f"Processed {total_reviews:,} reviews...")


# --------------------------------------------------
# 4. Save monthly results
# --------------------------------------------------

with open(MONTHLY_OUTPUT, "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)

    writer.writerow([
        "month",
        "total_reviews",
        "average_stars",
        "open_business_reviews",
        "closed_business_reviews",
        "open_average_stars",
        "closed_average_stars",
        "businesses_reviewed",
    ])

    for month in sorted(monthly):
        entry = monthly[month]

        def average(total, count):
            return round(total / count, 4) if count else ""

        writer.writerow([
            month,
            entry["reviews"],
            average(entry["rating_sum"], entry["rating_count"]),
            entry["open_reviews"],
            entry["closed_reviews"],
            average(
                entry["open_rating_sum"],
                entry["open_rating_count"],
            ),
            average(
                entry["closed_rating_sum"],
                entry["closed_rating_count"],
            ),
            len(entry["businesses_reviewed"]),
        ])


# --------------------------------------------------
# 5. Save restaurant-level timelines
# --------------------------------------------------

with open(BUSINESS_OUTPUT, "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)

    writer.writerow([
        "business_id",
        "is_open",
        "review_count",
        "first_review_date",
        "last_review_date",
        "average_review_stars",
        "undated_reviews",
    ])

    for business_id, stats in business_stats.items():
        writer.writerow([
            business_id,
            business_status[business_id],
            stats["review_count"],
            stats["first_review"] or "",
            stats["last_review"] or "",
            round(
                stats["rating_sum"] / stats["rating_count"], 4
            ) if stats["rating_count"] else "",
            stats["undated_reviews"],
        ])


# --------------------------------------------------
# 6. Print a concise report
# --------------------------------------------------

businesses_with_reviews = sum(
    stats["review_count"] > 0
    for stats in business_stats.values()
)

businesses_without_reviews = (
    len(business_stats) - businesses_with_reviews
)

print("\n" + "=" * 55)
print("EDA STEP 1: TEMPORAL COVERAGE")
print("=" * 55)

print(f"Total reviews scanned: {total_reviews:,}")
print(f"Missing business IDs: {missing_business_id:,}")
print(f"Unknown business IDs: {unknown_business_id:,}")
print(f"Missing review dates: {missing_date:,}")
print(f"Invalid review dates: {invalid_date:,}")
print(f"Reviews with missing/invalid stars: {missing_stars:,}")

print(f"\nEarliest review: {earliest_date}")
print(f"Latest review:   {latest_date}")

if earliest_date and latest_date:
    print(
        "Calendar span: "
        f"{(latest_date - earliest_date).days:,} days"
    )

print(f"\nRestaurants with reviews: {businesses_with_reviews:,}")
print(f"Restaurants without reviews: {businesses_without_reviews:,}")

for status, label in [(1, "OPEN"), (0, "CLOSED")]:
    subset = [
        stats for business_id, stats in business_stats.items()
        if business_status[business_id] == status
    ]

    with_reviews = sum(s["review_count"] > 0 for s in subset)
    print(
        f"{label}: {len(subset):,} restaurants; "
        f"{with_reviews:,} with reviews"
    )

print("\nOutput files:")
print(MONTHLY_OUTPUT)
print(BUSINESS_OUTPUT)
print("\nEDA Step 1 complete.")




