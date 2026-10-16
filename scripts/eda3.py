
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

# Load each business's current status.
status = {}
for business in read_jsonl_gz(BUSINESSES_FILE):
    status[business["business_id"]] = business["is_open"]

# Count reviews by business and calendar month.
monthly_counts = defaultdict(int)

for review in read_jsonl_gz(REVIEWS_FILE):
    business_id = review.get("business_id")
    raw_date = review.get("date")

    if business_id not in status or not raw_date:
        continue

    month = str(raw_date)[:7]  # YYYY-MM
    monthly_counts[(business_id, month)] += 1

# Find each business's last observed review month.
last_month = {}
for business_id, month in monthly_counts:
    if business_id not in last_month or month > last_month[business_id]:
        last_month[business_id] = month

# Compare activity in the 12 months leading up to each
# business's last observed review month.
rows = []

for business_id, end_month in last_month.items():
    end = pd.Period(end_month, freq="M")
    business_status = status[business_id]

    for months_before in range(12, -1, -1):
        month = end - months_before
        count = monthly_counts.get((business_id, str(month)), 0)

        rows.append({
            "business_id": business_id,
            "is_open": business_status,
            "months_before_last_review": -months_before,
            "reviews": count,
        })

df = pd.DataFrame(rows)

# Average review count per business per relative month.
summary = (
    df.groupby(["is_open", "months_before_last_review"])["reviews"]
    .mean()
    .reset_index()
)

summary.to_csv(
    DATA_DIR / "eda_relative_review_activity.csv",
    index=False
)

plt.figure(figsize=(11, 6))

for value, label in [(1, "Marked open"), (0, "Marked closed")]:
    subset = summary[summary["is_open"] == value]
    plt.plot(
        subset["months_before_last_review"],
        subset["reviews"],
        marker="o",
        markersize=3,
        label=label,
    )

plt.title("Average Review Activity Before Last Observed Review")
plt.xlabel("Months relative to last review (0 = last review month)")
plt.ylabel("Average reviews per business per month")
plt.xticks(range(-12, 1))
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "05_relative_review_activity.png",
    dpi=160
)
plt.show()

print("\nAverage reviews per business per month:")
print(summary.pivot(
    index="months_before_last_review",
    columns="is_open",
    values="reviews"
).rename(columns={0: "Marked closed", 1: "Marked open"}).round(3))

print("\nSaved: eda_relative_review_activity.csv")
print("Saved: 05_relative_review_activity.png")
