
import gzip
import json
from collections import Counter, defaultdict
from datetime import datetime
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
FEATURES_FILE = DATA_DIR / "business_features.csv"

SUMMARY_FILE = DATA_DIR / "review_audit_summary.txt"
YEARLY_FILE = DATA_DIR / "review_audit_by_year.csv"
RECONCILIATION_FILE = DATA_DIR / "review_audit_2021_reconciliation.csv"
ISSUES_FILE = DATA_DIR / "review_audit_issues.csv"

FEATURE_START = datetime(2021, 1, 1)
FEATURE_END = datetime(2022, 1, 1)

MAX_ISSUE_EXAMPLES = 1000


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def read_jsonl_gz(path):
    """Stream records from a compressed JSON Lines file."""
    with gzip.open(path, "rt", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                yield line_number, None
                continue

            yield line_number, record


def valid_date(raw_date):
    """Return a parsed date or None."""
    if not isinstance(raw_date, str):
        return None

    try:
        return datetime.strptime(raw_date[:10], "%Y-%m-%d")
    except ValueError:
        return None


def valid_rating(value):
    """Yelp review ratings should be numeric values from 1 to 5."""
    if isinstance(value, bool):
        return False

    return (
        isinstance(value, (int, float))
        and 1 <= value <= 5
    )


def add_issue(issues, issue_type, line_number, review_id=None,
              business_id=None, value=None):
    """Keep a limited number of examples of each issue."""
    if len(issues) >= MAX_ISSUE_EXAMPLES:
        return

    issues.append({
        "issue_type": issue_type,
        "line_number": line_number,
        "review_id": review_id,
        "business_id": business_id,
        "value": value,
    })


# --------------------------------------------------
# 1. Load the restaurant sample IDs
# --------------------------------------------------

print("Loading restaurant business IDs...")

business_ids = set()

for _, business in read_jsonl_gz(BUSINESSES_FILE):
    if business is None:
        continue

    business_id = business.get("business_id")

    if business_id:
        business_ids.add(business_id)

print(f"Restaurant business IDs: {len(business_ids):,}")


# --------------------------------------------------
# 2. Load existing business-level review counts
# --------------------------------------------------

features = pd.read_csv(
    FEATURES_FILE,
    usecols=["business_id", "reviews_2021"]
)

expected_2021 = dict(
    zip(features["business_id"], features["reviews_2021"])
)

print(f"Business feature rows: {len(expected_2021):,}")


# --------------------------------------------------
# 3. Initialize audit counters
# --------------------------------------------------

total_lines = 0
valid_json_records = 0
malformed_json_records = 0

missing_review_ids = 0
missing_business_ids = 0
missing_dates = 0
invalid_dates = 0
invalid_ratings = 0

reviews_outside_sample = 0
reviews_in_sample = 0

duplicate_review_ids = 0
duplicate_review_records = 0

# Exact duplicate detection requires remembering previously
# encountered IDs and record fingerprints.
seen_review_ids = set()
seen_record_fingerprints = set()

# Number of reviews per business during 2021.
actual_2021 = defaultdict(int)

# Review counts by calendar year.
yearly_counts = Counter()

# Other useful counts.
rating_counts = Counter()
reviews_with_text = 0
reviews_without_text = 0

issues = []

print("\nStarting review audit...")


# --------------------------------------------------
# 4. Stream through the review file
# --------------------------------------------------

for line_number, review in read_jsonl_gz(REVIEWS_FILE):
    total_lines += 1

    if review is None:
        malformed_json_records += 1

        add_issue(
            issues,
            "malformed_json",
            line_number
        )
        continue

    valid_json_records += 1

    review_id = review.get("review_id")
    business_id = review.get("business_id")
    raw_date = review.get("date")
    stars = review.get("stars")
    text = review.get("text")

    # Missing identifiers
    if not review_id:
        missing_review_ids += 1

        add_issue(
            issues,
            "missing_review_id",
            line_number,
            business_id=business_id
        )

    if not business_id:
        missing_business_ids += 1

        add_issue(
            issues,
            "missing_business_id",
            line_number,
            review_id=review_id
        )

    # Duplicate review IDs
    if review_id:
        if review_id in seen_review_ids:
            duplicate_review_ids += 1

            add_issue(
                issues,
                "duplicate_review_id",
                line_number,
                review_id=review_id,
                business_id=business_id
            )
        else:
            seen_review_ids.add(review_id)

    # Exact duplicate record detection.
    # Serialize the entire record consistently.
    fingerprint = json.dumps(
        review,
        sort_keys=True,
        ensure_ascii=False
    )

    if fingerprint in seen_record_fingerprints:
        duplicate_review_records += 1

        add_issue(
            issues,
            "exact_duplicate_record",
            line_number,
            review_id=review_id,
            business_id=business_id
        )
    else:
        seen_record_fingerprints.add(fingerprint)

    # Date validation
    if not raw_date:
        missing_dates += 1

        add_issue(
            issues,
            "missing_date",
            line_number,
            review_id=review_id,
            business_id=business_id
        )

        parsed_date = None
    else:
        parsed_date = valid_date(raw_date)

        if parsed_date is None:
            invalid_dates += 1

            add_issue(
                issues,
                "invalid_date",
                line_number,
                review_id=review_id,
                business_id=business_id,
                value=raw_date
            )

    # Rating validation
    if not valid_rating(stars):
        invalid_ratings += 1

        add_issue(
            issues,
            "invalid_rating",
            line_number,
            review_id=review_id,
            business_id=business_id,
            value=stars
        )
    else:
        rating_counts[stars] += 1

    # Review text
    if isinstance(text, str) and text.strip():
        reviews_with_text += 1
    else:
        reviews_without_text += 1

    # Restaurant sample membership
    if business_id not in business_ids:
        reviews_outside_sample += 1
        continue

    reviews_in_sample += 1

    # Yearly counts and 2021 counts
    if parsed_date is not None:
        yearly_counts[parsed_date.year] += 1

        if FEATURE_START <= parsed_date < FEATURE_END:
            actual_2021[business_id] += 1

    if total_lines % 500_000 == 0:
        print(f"Processed {total_lines:,} review records...")


# --------------------------------------------------
# 5. Compare actual 2021 counts with existing features
# --------------------------------------------------

print("\nBuilding 2021 reconciliation...")

reconciliation = features.copy()

reconciliation["actual_reviews_2021"] = (
    reconciliation["business_id"]
    .map(actual_2021)
    .fillna(0)
    .astype(int)
)

reconciliation["count_difference"] = (
    reconciliation["actual_reviews_2021"]
    - reconciliation["reviews_2021"]
)

reconciliation["matches"] = (
    reconciliation["count_difference"] == 0
)

reconciliation.to_csv(RECONCILIATION_FILE, index=False)


# --------------------------------------------------
# 6. Save yearly counts and issue examples
# --------------------------------------------------

yearly_df = pd.DataFrame(
    sorted(yearly_counts.items()),
    columns=["year", "review_count"]
)

yearly_df.to_csv(YEARLY_FILE, index=False)

pd.DataFrame(issues).to_csv(ISSUES_FILE, index=False)


# --------------------------------------------------
# 7. Generate summary
# --------------------------------------------------

matched = int(reconciliation["matches"].sum())
mismatched = len(reconciliation) - matched

summary = [
    "=" * 60,
    "YELP REVIEW DATA AUDIT",
    "=" * 60,
    "",
    f"Total lines: {total_lines:,}",
    f"Valid JSON records: {valid_json_records:,}",
    f"Malformed JSON records: {malformed_json_records:,}",
    "",
    f"Reviews with missing review IDs: {missing_review_ids:,}",
    f"Reviews with missing business IDs: {missing_business_ids:,}",
    f"Reviews with missing dates: {missing_dates:,}",
    f"Reviews with invalid dates: {invalid_dates:,}",
    f"Reviews with invalid ratings: {invalid_ratings:,}",
    "",
    f"Duplicate review IDs: {duplicate_review_ids:,}",
    f"Exact duplicate records: {duplicate_review_records:,}",
    "",
    f"Reviews outside restaurant sample: {reviews_outside_sample:,}",
    f"Reviews inside restaurant sample: {reviews_in_sample:,}",
    "",
    f"Reviews with text: {reviews_with_text:,}",
    f"Reviews without text: {reviews_without_text:,}",
    "",
    "2021 RECONCILIATION",
    "-" * 60,
    f"Businesses compared: {len(reconciliation):,}",
    f"Matching 2021 counts: {matched:,}",
    f"Mismatched 2021 counts: {mismatched:,}",
    "",
    "YEARLY REVIEW COUNTS",
    "-" * 60,
    yearly_df.to_string(index=False),
    "",
    "RATING COUNTS",
    "-" * 60,
    str(rating_counts),
    "",
    f"Issue examples saved: {len(issues):,}",
    "",
    "Output files:",
    str(SUMMARY_FILE),
    str(YEARLY_FILE),
    str(RECONCILIATION_FILE),
    str(ISSUES_FILE),
]

summary_text = "\n".join(summary)

SUMMARY_FILE.write_text(
    summary_text,
    encoding="utf-8"
)

print("\n" + summary_text)
print("\nReview audit complete.")
