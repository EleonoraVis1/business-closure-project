
import gzip
import json
from pathlib import Path
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = Path(
    r"C:\Users\zarni\Desktop\School\mining\business-closure-project\data"
)

BUSINESSES_FILE = DATA_DIR / "businesses.jsonl.gz"
REVIEWS_FILE = DATA_DIR / "reviews.jsonl.gz"
OUTPUT_DIR = DATA_DIR / "eda_charts"
OUTPUT_DIR.mkdir(exist_ok=True)


def read_jsonl_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


# --------------------------------------------------
# 1. Load current business status
# --------------------------------------------------

business_status = {}

for business in read_jsonl_gz(BUSINESSES_FILE):
    business_status[business["business_id"]] = business["is_open"]

print(f"Loaded {len(business_status):,} businesses.")
print("Scanning reviews...")


# --------------------------------------------------
# 2. Aggregate review data by business and month
# --------------------------------------------------

monthly = defaultdict(
    lambda: {
        "review_count": 0,
        "rating_sum": 0.0,
        "rating_count": 0,
        "reviewer_ids": set(),
    }
)

for i, review in enumerate(read_jsonl_gz(REVIEWS_FILE), start=1):
    business_id = review.get("business_id")
    review_date = review.get("date")
    reviewer_id = review.get("user_id")
    stars = review.get("stars")

    if business_id not in business_status or not review_date:
        continue

    month = str(review_date)[:7]
    entry = monthly[(business_id, month)]

    entry["review_count"] += 1

    if isinstance(stars, (int, float)) and 1 <= stars <= 5:
        entry["rating_sum"] += stars
        entry["rating_count"] += 1

    if reviewer_id:
        entry["reviewer_ids"].add(reviewer_id)

    if i % 500_000 == 0:
        print(f"Processed {i:,} reviews...")


# --------------------------------------------------
# 3. Find each business's last observed review month
# --------------------------------------------------

last_month = {}

for business_id, month in monthly:
    if (
        business_id not in last_month
        or month > last_month[business_id]
    ):
        last_month[business_id] = month


# --------------------------------------------------
# 4. Create a relative timeline for each business
# --------------------------------------------------

rows = []

for business_id, end_month in last_month.items():
    end = pd.Period(end_month, freq="M")
    status = business_status[business_id]

    for months_before in range(12, -1, -1):
        month = str(end - months_before)
        entry = monthly.get((business_id, month))

        if entry:
            review_count = entry["review_count"]
            rating_count = entry["rating_count"]

            average_stars = (
                entry["rating_sum"] / rating_count
                if rating_count else None
            )

            distinct_reviewers = len(entry["reviewer_ids"])

            # Fraction of reviews represented by distinct
            # reviewers. Higher values mean fewer repeat
            # reviews by the same users in that month.
            reviewer_diversity = (
                distinct_reviewers / review_count
                if review_count else None
            )
        else:
            review_count = 0
            average_stars = None
            distinct_reviewers = 0
            reviewer_diversity = None

        rows.append({
            "business_id": business_id,
            "is_open": status,
            "months_before_last_review": -months_before,
            "calendar_month": month,
            "review_count": review_count,
            "average_stars": average_stars,
            "distinct_reviewers": distinct_reviewers,
            "reviewer_diversity": reviewer_diversity,
        })

df = pd.DataFrame(rows)


# --------------------------------------------------
# 5. Summarize each relative month by current status
# --------------------------------------------------

summary = (
    df.groupby(["is_open", "months_before_last_review"])
    .agg(
        mean_reviews=("review_count", "mean"),
        mean_stars=("average_stars", "mean"),
        mean_distinct_reviewers=("distinct_reviewers", "mean"),
        mean_reviewer_diversity=("reviewer_diversity", "mean"),
        businesses=("business_id", "nunique"),
        businesses_with_ratings=("average_stars", "count"),
    )
    .reset_index()
)

summary.to_csv(
    DATA_DIR / "eda_rating_reviewer_summary.csv",
    index=False,
)

df.to_csv(
    DATA_DIR / "eda_business_monthly_features.csv",
    index=False,
)


# --------------------------------------------------
# 6. Plot average ratings
# --------------------------------------------------

plt.figure(figsize=(10, 5))

for status, label in [(1, "Marked open"), (0, "Marked closed")]:
    subset = summary[summary["is_open"] == status]
    plt.plot(
        subset["months_before_last_review"],
        subset["mean_stars"],
        marker="o",
        label=label,
    )

plt.title("Average Rating Before Last Observed Review")
plt.xlabel("Months before last review")
plt.ylabel("Average stars")
plt.xticks(range(-12, 1))
plt.ylim(1, 5)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "06_rating_trends.png", dpi=160)
plt.show()


# --------------------------------------------------
# 7. Plot distinct reviewers
# --------------------------------------------------

plt.figure(figsize=(10, 5))

for status, label in [(1, "Marked open"), (0, "Marked closed")]:
    subset = summary[summary["is_open"] == status]
    plt.plot(
        subset["months_before_last_review"],
        subset["mean_distinct_reviewers"],
        marker="o",
        label=label,
    )

plt.title("Distinct Reviewers Before Last Observed Review")
plt.xlabel("Months before last review")
plt.ylabel("Mean distinct reviewers per business per month")
plt.xticks(range(-12, 1))
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "07_distinct_reviewers.png", dpi=160)
plt.show()


# --------------------------------------------------
# 8. Plot reviewer diversity
# --------------------------------------------------

plt.figure(figsize=(10, 5))

for status, label in [(1, "Marked open"), (0, "Marked closed")]:
    subset = summary[summary["is_open"] == status]
    plt.plot(
        subset["months_before_last_review"],
        subset["mean_reviewer_diversity"],
        marker="o",
        label=label,
    )

plt.title("Reviewer Diversity Before Last Observed Review")
plt.xlabel("Months before last review")
plt.ylabel("Mean distinct reviewers / reviews")
plt.xticks(range(-12, 1))
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "08_reviewer_diversity.png", dpi=160)
plt.show()


# --------------------------------------------------
# 9. Print results
# --------------------------------------------------

print("\nEDA STEP 3: RATINGS AND REVIEWER ACTIVITY")
print("=" * 55)

for status, label in [(0, "MARKED CLOSED"), (1, "MARKED OPEN")]:
    subset = summary[summary["is_open"] == status].sort_values(
        "months_before_last_review"
    )

    print(f"\n{label}")
    print(subset[
        [
            "months_before_last_review",
            "mean_reviews",
            "mean_stars",
            "mean_distinct_reviewers",
            "mean_reviewer_diversity",
            "businesses_with_ratings",
        ]
    ].round(3).to_string(index=False))

print("\nSaved:")
print(DATA_DIR / "eda_rating_reviewer_summary.csv")
print(DATA_DIR / "eda_business_monthly_features.csv")
print(f"Charts: {OUTPUT_DIR}")
