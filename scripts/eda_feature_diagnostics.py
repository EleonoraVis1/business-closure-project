
from pathlib import Path
import pandas as pd

DATA_DIR = Path(
    r"C:\Users\zarni\Desktop\School\mining\business-closure-project\data"
)

INPUT_FILE = DATA_DIR / "business_features.csv"
OUTPUT_FILE = DATA_DIR / "business_feature_diagnostics.csv"

df = pd.read_csv(INPUT_FILE)

# Indicators preserve the distinction between no data and low ratings.
df["has_reviews_2021"] = (df["reviews_2021"] > 0).astype(int)
df["has_reviews_recent_3m"] = (
    df["reviews_recent_3m_2021"] > 0
).astype(int)
df["has_ratings_both_halves"] = (
    df["avg_review_stars_first_6m"].notna()
    & df["avg_review_stars_recent_6m"].notna()
).astype(int)

print("=" * 65)
print("FEATURE DIAGNOSTICS")
print("=" * 65)

print("\n1. Zero recent activity by status")
for status, label in [(0, "Marked closed"), (1, "Marked open")]:
    group = df[df["is_open"] == status]
    print(f"\n{label} (n={len(group):,})")
    for column in ["reviews_2021", "reviews_recent_3m_2021", "tips_2021"]:
        zero_pct = (group[column] == 0).mean() * 100
        print(f"  {column}: {zero_pct:.1f}% have zero")

print("\n2. Missing ratings by status")
missing_summary = df.groupby("is_open").agg(
    businesses=("business_id", "count"),
    missing_2021_rating=("avg_review_stars_2021",
                          lambda s: s.isna().sum()),
    missing_rating_change=("rating_change_recent_vs_first_6m",
                           lambda s: s.isna().sum()),
    with_ratings_both_halves=("has_ratings_both_halves", "sum"),
)
print(missing_summary.to_string())

print("\n3. Feature distribution by status")
features = [
    "business_stars",
    "business_review_count",
    "reviews_2021",
    "reviews_recent_3m_2021",
    "avg_review_stars_2021",
    "rating_change_recent_vs_first_6m",
    "all_time_checkins",
    "tips_2021",
]

for status, label in [(0, "Marked closed"), (1, "Marked open")]:
    print(f"\n{label}")
    print(
        df.loc[df["is_open"] == status, features]
        .quantile([0.00, 0.25, 0.50, 0.75, 0.90, 0.99])
        .round(3)
        .to_string()
    )

# Save indicators with the original feature table.
df.to_csv(OUTPUT_FILE, index=False)

print(f"\nSaved diagnostics table to: {OUTPUT_FILE}")
