
import gzip
import json
import os
from pathlib import Path


# =========================================================
# Configuration — EDIT THESE PATHS
# =========================================================

SOURCE_DIR = Path(r"C:\Users\YourUser\Downloads\yelp_dataset")
OUTPUT_DIR = Path(r"C:\Users\YourUser\Documents\business_closure\data")

BUSINESS_FILE = SOURCE_DIR / "business.json"
REVIEW_FILE = SOURCE_DIR / "review.json"
CHECKIN_FILE = SOURCE_DIR / "checkin.json"
TIP_FILE = SOURCE_DIR / "tip.json"

# Keep businesses whose category list includes "Restaurants".
# This is deliberately narrower than all businesses in Food.
RESTAURANT_CATEGORY = "Restaurants"

# Preserve review text for possible sentiment analysis.
KEEP_TIPS = True


# =========================================================
# Helpers
# =========================================================

def read_json_lines(path):
    """Read a large JSON Lines file one record at a time."""
    with open(path, "rt", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue

            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                print(f"WARNING: Invalid JSON at {path.name}, "
                      f"line {line_number}")


def write_json_gz(path, records):
    """Write records as compressed JSON Lines."""
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(
                record, ensure_ascii=False, separators=(",", ":")
            ))
            f.write("\n")


def filtered_output_path(name):
    return OUTPUT_DIR / f"{name}.jsonl.gz"


def filter_to_gzip(input_path, output_path, keep_ids, fields):
    """Filter one large source file without loading it all into RAM."""
    count = 0
    matched = 0
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    try:
        with open(input_path, "rt", encoding="utf-8") as source, \
             gzip.open(temp_path, "wt", encoding="utf-8") as target:

            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    print(f"WARNING: Invalid JSON at "
                          f"{input_path.name}, line {line_number}")
                    continue

                count += 1

                if record.get("business_id") not in keep_ids:
                    continue

                reduced = {
                    field: record[field]
                    for field in fields
                    if field in record
                }

                target.write(json.dumps(
                    reduced, ensure_ascii=False, separators=(",", ":")
                ))
                target.write("\n")
                matched += 1

                if count % 500_000 == 0:
                    print(f"{input_path.name}: read {count:,}, "
                          f"retained {matched:,}")

        # Publish the output only after the stage finishes successfully.
        os.replace(temp_path, output_path)

    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise

    print(f"Finished {input_path.name}: "
          f"{count:,} records read, {matched:,} retained")
    print(f"Output: {output_path}")
    print(f"Compressed size: {output_path.stat().st_size / 1024**2:.2f} MB")

    return matched


# =========================================================
# Stage 1: Select restaurant businesses
# =========================================================

def filter_businesses():
    output_path = filtered_output_path("businesses")
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    business_ids = set()
    total = 0
    retained = 0

    business_fields = [
        "business_id", "name", "address", "city", "state",
        "postal_code", "latitude", "longitude", "stars",
        "review_count", "is_open", "attributes", "categories",
        "hours"
    ]

    try:
        with open(BUSINESS_FILE, "rt", encoding="utf-8") as source, \
             gzip.open(temp_path, "wt", encoding="utf-8") as target:

            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    print(f"WARNING: Invalid business JSON at line "
                          f"{line_number}")
                    continue

                total += 1

                categories = record.get("categories") or ""
                category_set = {
                    category.strip()
                    for category in categories.split(",")
                }

                if RESTAURANT_CATEGORY not in category_set:
                    continue

                business_id = record.get("business_id")
                if not business_id:
                    continue

                business_ids.add(business_id)

                reduced = {
                    field: record[field]
                    for field in business_fields
                    if field in record
                }

                target.write(json.dumps(
                    reduced, ensure_ascii=False, separators=(",", ":")
                ))
                target.write("\n")
                retained += 1

                if total % 100_000 == 0:
                    print(f"Businesses scanned: {total:,}; "
                          f"restaurants found: {retained:,}")

        os.replace(temp_path, output_path)

    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise

    print(f"\nBusinesses scanned: {total:,}")
    print(f"Restaurants retained: {retained:,}")
    print(f"Saved to: {output_path}")
    print(f"Compressed size: {output_path.stat().st_size / 1024**2:.2f} MB")

    return business_ids


# =========================================================
# Main pipeline
# =========================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for path in [BUSINESS_FILE, REVIEW_FILE, CHECKIN_FILE]:
        if not path.is_file():
            raise FileNotFoundError(f"Input file not found: {path}")

    if KEEP_TIPS and not TIP_FILE.is_file():
        print("Tips file not found; tips will be skipped.")

    print("=" * 60)
    print("STAGE 1 — FILTER BUSINESSES")
    print("=" * 60)

    restaurant_ids = filter_businesses()

    if not restaurant_ids:
        raise RuntimeError(
            "No restaurants were found. Check the input file and "
            "category filter before proceeding."
        )

    print(f"\nRestaurant IDs held in memory: {len(restaurant_ids):,}")

    print("\n" + "=" * 60)
    print("STAGE 2 — FILTER REVIEWS")
    print("=" * 60)

    review_fields = [
        "review_id", "business_id", "user_id", "date", "stars",
        "text", "useful", "funny", "cool"
    ]

    filter_to_gzip(
        REVIEW_FILE,
        filtered_output_path("reviews"),
        restaurant_ids,
        review_fields
    )

    print("\n" + "=" * 60)
    print("STAGE 3 — FILTER CHECK-INS")
    print("=" * 60)

    filter_to_gzip(
        CHECKIN_FILE,
        filtered_output_path("checkins"),
        restaurant_ids,
        ["business_id", "date"]
    )

    if KEEP_TIPS and TIP_FILE.is_file():
        print("\n" + "=" * 60)
        print("STAGE 4 — FILTER TIPS")
        print("=" * 60)

        filter_to_gzip(
            TIP_FILE,
            filtered_output_path("tips"),
            restaurant_ids,
            ["user_id", "business_id", "text", "date",
             "compliment_count"]
        )

    print("\n" + "=" * 60)
    print("FILTERING COMPLETE")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_DIR}")
    print("Original files have NOT been deleted.")


if __name__ == "__main__":
    main()
