
import gzip
import json
from pathlib import Path
from collections import Counter

DATA_DIR = Path(r"C:\Users\zarni\Desktop\School\mining")

BUSINESSES = DATA_DIR / "businesses.jsonl.gz"
REVIEWS = DATA_DIR / "reviews.jsonl.gz"
CHECKINS = DATA_DIR / "checkins.jsonl.gz"
TIPS = DATA_DIR / "tips.jsonl.gz"


def read_jsonl(path):
    """Read a compressed JSON Lines file one record at a time."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def main():
    print("=" * 65)
    print("YELP DATASET VALIDATION")
    print("=" * 65)

    # -----------------------------------------------------
    # 1. Read businesses and inspect closure labels
    # -----------------------------------------------------
    business_ids = set()
    status_counts = Counter()
    missing_status = 0
    invalid_status = 0
    duplicate_business_ids = 0
    missing_business_ids = 0

    open_stars = []
    closed_stars = []
    open_review_counts = []
    closed_review_counts = []

    total_businesses = 0

    print("\n[1/5] Reading businesses...")

    for business in read_jsonl(BUSINESSES):
        total_businesses += 1
        business_id = business.get("business_id")

        if not business_id:
            missing_business_ids += 1
            continue

        if business_id in business_ids:
            duplicate_business_ids += 1
        business_ids.add(business_id)

        status = business.get("is_open")

        if status is None or status == "":
            missing_status += 1
            continue

        if status not in (0, 1, False, True):
            invalid_status += 1
            status_counts[str(status)] += 1
            continue

        status = int(status)
        status_counts[status] += 1

        stars = business.get("stars")
        review_count = business.get("review_count")

        if status == 1:
            if isinstance(stars, (int, float)):
                open_stars.append(stars)
            if isinstance(review_count, (int, float)):
                open_review_counts.append(review_count)
        else:
            if isinstance(stars, (int, float)):
                closed_stars.append(stars)
            if isinstance(review_count, (int, float)):
                closed_review_counts.append(review_count)

    print(f"Business records: {total_businesses:,}")
    print(f"Unique business IDs: {len(business_ids):,}")
    print(f"Duplicate business IDs: {duplicate_business_ids:,}")
    print(f"Missing business IDs: {missing_business_ids:,}")
    print(f"Missing is_open values: {missing_status:,}")
    print(f"Unexpected is_open values: {invalid_status:,}")
    print(f"Status counts: {dict(status_counts)}")

    # -----------------------------------------------------
    # 2. Check references in reviews, check-ins, and tips
    # -----------------------------------------------------
    def check_references(label, path):
        total = 0
        missing_id = 0
        unmatched = 0

        print(f"\nChecking {label}...")

        for record in read_jsonl(path):
            total += 1
            business_id = record.get("business_id")

            if not business_id:
                missing_id += 1
            elif business_id not in business_ids:
                unmatched += 1

            if total % 500_000 == 0:
                print(f"  Checked {total:,} {label}...")

        print(f"  Total records: {total:,}")
        print(f"  Missing business_id: {missing_id:,}")
        print(f"  No matching restaurant: {unmatched:,}")

        return total, missing_id, unmatched

    print("\n[2/5] Checking reviews...")
    review_results = check_references("reviews", REVIEWS)

    print("\n[3/5] Checking check-ins...")
    checkin_results = check_references("check-ins", CHECKINS)

    print("\n[4/5] Checking tips...")
    tip_results = check_references("tips", TIPS)

    # -----------------------------------------------------
    # 3. Compare summary statistics by closure label
    # -----------------------------------------------------
    def average(values):
        return sum(values) / len(values) if values else None

    print("\n[5/5] Closure-label summary")
    print("-" * 65)

    for label, count, stars, reviews in [
        ("OPEN", status_counts[1], open_stars, open_review_counts),
        ("CLOSED", status_counts[0], closed_stars, closed_review_counts),
    ]:
        print(f"\n{label} businesses: {count:,}")
        print(f"  Average stars: {average(stars)}")
        print(f"  Average business review_count: {average(reviews)}")

    print("\n" + "=" * 65)
    print("VALIDATION SUMMARY")
    print("=" * 65)

    all_results = [
        ("Reviews", review_results),
        ("Check-ins", checkin_results),
        ("Tips", tip_results),
    ]

    all_references_match = all(
        missing == 0 and unmatched == 0
        for _, (_, missing, unmatched) in all_results
    )

    labels_valid = (
        missing_status == 0
        and invalid_status == 0
        and status_counts[0] + status_counts[1]
        == len(business_ids)
    )

    print(
        "Referential integrity:",
        "PASS" if all_references_match else "CHECK NEEDED",
    )
    print(
        "Closure labels:",
        "PASS" if labels_valid else "CHECK NEEDED",
    )

    if duplicate_business_ids or missing_business_ids:
        print("WARNING: Business ID quality issues were detected.")

    print("\nValidation complete. No files were modified.")


if __name__ == "__main__":
    main()
