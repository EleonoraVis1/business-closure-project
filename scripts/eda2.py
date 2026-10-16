
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path(
    r"C:\Users\zarni\Desktop\School\mining\business-closure-project\data"
)

INPUT_FILE = DATA_DIR / "eda_monthly_reviews.csv"
OUTPUT_DIR = DATA_DIR / "eda_charts"
OUTPUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(INPUT_FILE, parse_dates=["month"])
df = df.sort_values("month")

# Remove any incomplete rows in the primary monthly series.
df = df.dropna(subset=["month", "total_reviews"])

# --------------------------------------------------
# Chart 1: Total reviews per month
# --------------------------------------------------

plt.figure(figsize=(13, 5))
plt.plot(df["month"], df["total_reviews"], linewidth=1)
plt.title("Monthly Yelp Restaurant Reviews")
plt.xlabel("Month")
plt.ylabel("Number of reviews")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "01_monthly_review_volume.png", dpi=160)
plt.show()

# --------------------------------------------------
# Chart 2: Monthly reviews by current business status
# --------------------------------------------------

plt.figure(figsize=(13, 5))
plt.plot(
    df["month"],
    df["open_business_reviews"],
    label="Businesses marked open",
    linewidth=1,
)
plt.plot(
    df["month"],
    df["closed_business_reviews"],
    label="Businesses marked closed",
    linewidth=1,
)
plt.title("Monthly Reviews by Current Business Status")
plt.xlabel("Month")
plt.ylabel("Number of reviews")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "02_reviews_by_status.png", dpi=160)
plt.show()

# --------------------------------------------------
# Chart 3: Average review rating over time
# --------------------------------------------------

plt.figure(figsize=(13, 5))
plt.plot(
    df["month"],
    df["average_stars"],
    label="All businesses",
    linewidth=1,
)
plt.plot(
    df["month"],
    df["open_average_stars"],
    label="Businesses marked open",
    linewidth=1,
)
plt.plot(
    df["month"],
    df["closed_average_stars"],
    label="Businesses marked closed",
    linewidth=1,
)
plt.title("Monthly Average Review Rating")
plt.xlabel("Month")
plt.ylabel("Average stars")
plt.ylim(1, 5)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "03_monthly_average_ratings.png", dpi=160)
plt.show()

# --------------------------------------------------
# Chart 4: Review volume, smoothed over 12 months
# --------------------------------------------------

df["reviews_12_month_avg"] = (
    df["total_reviews"].rolling(window=12, min_periods=6).mean()
)

plt.figure(figsize=(13, 5))
plt.plot(
    df["month"],
    df["total_reviews"],
    label="Monthly reviews",
    alpha=0.4,
)
plt.plot(
    df["month"],
    df["reviews_12_month_avg"],
    label="12-month rolling average",
    linewidth=2,
)
plt.title("Monthly Review Volume and 12-Month Trend")
plt.xlabel("Month")
plt.ylabel("Number of reviews")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "04_review_volume_trend.png", dpi=160)
plt.show()

# --------------------------------------------------
# Print summary statistics
# --------------------------------------------------

print("\nEDA STEP 2: MONTHLY TRENDS")
print("=" * 45)

print(f"Months represented: {len(df)}")

if not df.empty:
    print(f"First month: {df['month'].min():%Y-%m}")
    print(f"Last month:  {df['month'].max():%Y-%m}")

    peak = df.loc[df["total_reviews"].idxmax()]
    quiet = df.loc[df["total_reviews"].idxmin()]

    print(
        f"\nHighest review month: {peak['month']:%Y-%m} "
        f"({int(peak['total_reviews']):,} reviews)"
    )
    print(
        f"Lowest review month: {quiet['month']:%Y-%m} "
        f"({int(quiet['total_reviews']):,} reviews)"
    )

    print(
        "\nAverage monthly review count: "
        f"{df['total_reviews'].mean():,.1f}"
    )

print(f"\nCharts saved to: {OUTPUT_DIR}")
