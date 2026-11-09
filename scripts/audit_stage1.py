
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(
    r"C:\Users\zarni\Desktop\School\mining\business-closure-project"
)
DATA_DIR = PROJECT_DIR / "data"

FILES = {
    "features": DATA_DIR / "business_features.csv",
    "temporal": DATA_DIR / "eda_temporal_audit.csv",
    "diagnostics": DATA_DIR / "business_features_diagnostics.csv",
}

# --------------------------------------------------
# 1. Load files
# --------------------------------------------------

dfs = {}

for name, path in FILES.items():
    if path.exists():
        dfs[name] = pd.read_csv(path)
        print(f"\n{name}: {len(dfs[name]):,} rows, "
              f"{len(dfs[name].columns)} columns")
    else:
        print(f"\nMissing file: {path}")

if "features" not in dfs:
    raise FileNotFoundError("business_features.csv was not found.")

df = dfs["features"]

# --------------------------------------------------
# 2. Business ID and target validation
# --------------------------------------------------

print("\n" + "=" * 60)
print("BUSINESS AND TARGET VALIDATION")
print("=" * 60)

print("Rows:", len(df))
print("Unique business IDs:", df["business_id"].nunique())
print("Duplicate business IDs:", df["business_id"].duplicated().sum())
print("Missing business IDs:", df["business_id"].isna().sum())

print("\nStatus counts:")
print(df["is_open"].value_counts(dropna=False).sort_index())

print("\nInvalid status values:")
print(df.loc[~df["is_open"].isin([0, 1]), "is_open"].value_counts())

# --------------------------------------------------
# 3. Missing-value report
# --------------------------------------------------

missing = pd.DataFrame({
    "missing_count": df.isna().sum(),
    "missing_percent": df.isna().mean() * 100,
    "non_missing_count": df.notna().sum(),
})

missing = missing.sort_values(
    "missing_percent", ascending=False
)

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)
print(missing.round(2).to_string())

missing.to_csv(DATA_DIR / "stage1_missing_values.csv")

# Missing values by recorded status
if df["is_open"].notna().any():
    missing_by_status = (
        df.groupby("is_open")
        .apply(lambda group: group.isna().mean() * 100,
               include_groups=False)
        .T
    )
    missing_by_status.columns = [
        f"status_{column}" for column in missing_by_status.columns
    ]
    missing_by_status.to_csv(
        DATA_DIR / "stage1_missing_by_status.csv"
    )

# --------------------------------------------------
# 4. Numeric feature validation
# --------------------------------------------------

numeric_columns = [
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

for column in numeric_columns:
    if column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

print("\n" + "=" * 60)
print("NUMERIC VALIDATION")
print("=" * 60)

for column in numeric_columns:
    if column not in df.columns:
        continue

    print(f"\n{column}")
    print(df[column].describe().round(3).to_string())

# Ratings must be between 1 and 5
rating_columns = [
    "business_stars",
    "avg_review_stars_2021",
    "avg_review_stars_first_6m",
    "avg_review_stars_recent_6m",
]

for column in rating_columns:
    if column in df.columns:
        invalid = df[column].notna() & ~df[column].between(1, 5)
        print(f"\nInvalid values in {column}: {invalid.sum()}")

        if invalid.any():
            print(df.loc[invalid, ["business_id", column]].head(10))

# Counts must be nonnegative
count_columns = [
    "business_review_count",
    "reviews_2021",
    "reviews_first_6m_2021",
    "reviews_recent_6m_2021",
    "reviews_recent_3m_2021",
    "all_time_checkins",
    "tips_2021",
    "distinct_tip_users_2021",
]

for column in count_columns:
    if column in df.columns:
        invalid = df[column].notna() & (df[column] < 0)
        print(f"Negative values in {column}: {invalid.sum()}")

# --------------------------------------------------
# 5. Review count consistency
# --------------------------------------------------

required = {
    "reviews_2021",
    "reviews_first_6m_2021",
    "reviews_recent_6m_2021",
    "reviews_recent_3m_2021",
}

if required.issubset(df.columns):
    print("\n" + "=" * 60)
    print("REVIEW COUNT CONSISTENCY")
    print("=" * 60)

    mismatch = (
        df["reviews_2021"]
        != df["reviews_first_6m_2021"]
        + df["reviews_recent_6m_2021"]
    )

    print("2021 count != first half + second half:", mismatch.sum())

    invalid_recent = (
        df["reviews_recent_3m_2021"]
        > df["reviews_recent_6m_2021"]
    )

    print("Recent 3-month count > recent 6-month count:",
          invalid_recent.sum())

# --------------------------------------------------
# 6. Compare the same businesses across files
# --------------------------------------------------

if "temporal" in dfs:
    temporal = dfs["temporal"]

    print("\n" + "=" * 60)
    print("COMPARING FEATURE AND TEMPORAL FILES")
    print("=" * 60)

    common_ids = set(df["business_id"]) & set(temporal["business_id"])
    print("Business IDs in both files:", len(common_ids))

    compare_columns = [
        "avg_review_stars_recent_6m",
        "rating_change_recent_vs_first_6m",
        "all_time_checkins",
        "tips_2021",
    ]

    left = df.set_index("business_id")
    right = temporal.set_index("business_id")

    for column in compare_columns:
        if column not in left.columns or column not in right.columns:
            continue

        a = pd.to_numeric(left.loc[list(common_ids), column],
                          errors="coerce")
        b = pd.to_numeric(right.loc[list(common_ids), column],
                          errors="coerce")

        both_present = a.notna() & b.notna()
        different = both_present & (a != b)

        print(f"\n{column}")
        print("Both values present:", both_present.sum())
        print("Different values:", different.sum())

        if different.any():
            print(pd.DataFrame({
                "features": a[different],
                "temporal": b[different],
            }).head(10))

# --------------------------------------------------
# 7. Save final audit summary
# --------------------------------------------------

print("\n" + "=" * 60)
print("AUDIT COMPLETE")
print("=" * 60)

print("Missing-value report saved.")
print("Missing-by-status report saved.")
