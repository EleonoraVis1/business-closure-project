"""Small-sample Yelp closure validation using the free hospitality WARN CSV.

Run from the project root:
    python scripts/validate_closures_warn.py

Expected inputs:
    data/business_features.csv
    data/businesses.jsonl.gz

Outputs:
    data/closure_validation_sample.csv
    data/warn_candidate_matches.csv
    data/wayback_review_queue.csv

Every WARN match is a candidate requiring manual verification.
"""
from pathlib import Path
import difflib
import re
import urllib.parse
import urllib.request

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FEATURES = DATA / "business_features.csv"
BUSINESSES = DATA / "businesses.jsonl.gz"
WARN_PATH = DATA / "hospitality_layoff_notices.csv"
WARN_URL = (
    "https://huggingface.co/datasets/APProjects/"
    "us-restaurant-hotel-closings-layoffs-warn-act-notices-daily/"
    "resolve/main/data/hospitality_layoff_notices.csv"
)
PER_GROUP = 15
SEED = 20261018

DROP_WORDS = {
    "inc", "incorporated", "llc", "ltd", "limited", "corp", "corporation",
    "company", "co", "the"
}


def clean(value):
    if pd.isna(value):
        return ""
    value = str(value).lower().replace("&", " and ")
    words = re.sub(r"[^a-z0-9]+", " ", value).split()
    return " ".join(w for w in words if w not in DROP_WORDS)


def similarity(a, b):
    a, b = clean(a), clean(b)
    if not a or not b:
        return 0.0
    seq = difflib.SequenceMatcher(None, a, b).ratio()
    aa, bb = set(a.split()), set(b.split())
    overlap = len(aa & bb) / max(1, len(aa | bb))
    return max(seq, overlap)


def city_overlap(city, location):
    city, location = clean(city), clean(location)
    return bool(city and location and (city in location or location in city))


def main():
    DATA.mkdir(exist_ok=True)
    if not FEATURES.exists() or not BUSINESSES.exists():
        raise FileNotFoundError(
            f"Expected both {FEATURES} and {BUSINESSES}. "
            "Check the filenames in your data folder."
        )

    features = pd.read_csv(FEATURES)
    businesses = pd.read_json(BUSINESSES, lines=True, compression="gzip")
    required = {"business_id", "is_open", "city", "state"}
    if not required.issubset(features.columns):
        raise ValueError(f"business_features.csv needs columns: {sorted(required)}")
    if not {"business_id", "name"}.issubset(businesses.columns):
        raise ValueError("businesses.jsonl.gz needs business_id and name columns")

    business_cols = ["business_id", "name"]
    if "address" in businesses.columns:
        business_cols.append("address")
    df = features.merge(businesses[business_cols], on="business_id", how="left")
    df = df.dropna(subset=["name"]).copy()
    activity_cols = [c for c in [
        "reviews_2021", "reviews_recent_3m_2021",
        "business_review_count", "all_time_checkins"
    ] if c in df.columns]
    df["_activity"] = df[activity_cols].fillna(0).sum(axis=1) if activity_cols else 0

    # Balanced by current snapshot status, with both lower- and higher-activity
    # businesses represented within each status.
    pieces = []
    for status in [0, 1]:
        group = df[df["is_open"].astype(int) == status].copy()
        if group.empty:
            continue
        median = group["_activity"].median()
        for activity_group, mask in [
            ("lower_activity", group["_activity"] <= median),
            ("higher_activity", group["_activity"] > median),
        ]:
            subset = group[mask]
            n = min(PER_GROUP, len(subset))
            if n:
                picked = subset.sample(n=n, random_state=SEED)
                picked["status_label"] = "open_snapshot" if status else "closed_snapshot"
                picked["activity_stratum"] = activity_group
                pieces.append(picked)
    sample = pd.concat(pieces, ignore_index=True)
    sample.to_csv(DATA / "closure_validation_sample.csv", index=False)

    if not WARN_PATH.exists():
        print("Downloading WARN CSV...")
        urllib.request.urlretrieve(WARN_URL, WARN_PATH)
    warn = pd.read_csv(WARN_PATH)
    required_warn = {"state", "employer", "notice_date", "effective_date",
                     "notice_type", "location"}
    if not required_warn.issubset(warn.columns):
        raise ValueError(f"WARN CSV missing columns: {sorted(required_warn - set(warn.columns))}")

    rows = []
    for _, b in sample.iterrows():
        state = str(b.get("state", "")).strip().upper()
        if not state or state == "NAN":
            continue
        possible = warn[warn["state"].astype(str).str.upper() == state]
        for _, w in possible.iterrows():
            score = similarity(b["name"], w["employer"])
            city_match = city_overlap(b.get("city", ""), w.get("location", ""))
            # Conservative candidate filter. This does not establish identity.
            if score < 0.72 or (score < 0.82 and not city_match):
                continue
            if score >= 0.82 and city_match:
                tier = "strong_candidate_review_required"
            elif city_match:
                tier = "possible_candidate_review_required"
            else:
                tier = "name_match_location_unconfirmed"
            rows.append({
                "business_id": b["business_id"],
                "yelp_name": b["name"],
                "yelp_city": b.get("city", ""),
                "yelp_state": state,
                "yelp_is_open_snapshot": b.get("is_open", ""),
                "warn_employer": w.get("employer", ""),
                "warn_location": w.get("location", ""),
                "warn_state": w.get("state", ""),
                "warn_notice_date": w.get("notice_date", ""),
                "warn_effective_date": w.get("effective_date", ""),
                "warn_notice_type": w.get("notice_type", ""),
                "warn_employees_affected": w.get("employees_affected", ""),
                "name_similarity": round(score, 3),
                "city_location_overlap": city_match,
                "candidate_tier": tier,
                "manual_decision": "",
                "evidence_url_or_note": "",
            })

    columns = [
        "business_id", "yelp_name", "yelp_city", "yelp_state",
        "yelp_is_open_snapshot", "warn_employer", "warn_location", "warn_state",
        "warn_notice_date", "warn_effective_date", "warn_notice_type",
        "warn_employees_affected", "name_similarity", "city_location_overlap",
        "candidate_tier", "manual_decision", "evidence_url_or_note"
    ]
    matches = pd.DataFrame(rows, columns=columns)
    if not matches.empty:
        matches = matches.sort_values("name_similarity", ascending=False)
        matches = matches.drop_duplicates(
            ["business_id", "warn_employer", "warn_effective_date"]
        )
    matches.to_csv(DATA / "warn_candidate_matches.csv", index=False)

    strong_ids = set(
        matches.loc[matches["candidate_tier"] == "strong_candidate_review_required",
                    "business_id"]
    ) if not matches.empty else set()
    queue = sample.copy()
    queue["has_strong_warn_candidate"] = queue["business_id"].isin(strong_ids)
    queue["wayback_priority"] = queue["has_strong_warn_candidate"].map(
        {True: "cross_check", False: "prioritize"}
    )
    queue["search_query"] = queue.apply(
        lambda r: urllib.parse.quote(
            f'"{r["name"]}" "{r.get("city", "")}" {r.get("state", "")} '
            f'(closed OR "permanently closed" OR closure)'
        ), axis=1
    )
    queue["google_search_url"] = queue["search_query"].map(
        lambda q: "https://www.google.com/search?q=" + q
    )
    queue["wayback_machine_url"] = "https://web.archive.org/"
    queue.to_csv(DATA / "wayback_review_queue.csv", index=False)

    print("\n=== WARN pilot summary ===")
    print(f"Yelp businesses sampled: {len(sample)}")
    print(sample.groupby(["status_label", "activity_stratum"]).size().to_string())
    print(f"WARN rows: {len(warn):,}")
    print(f"Candidate Yelp-WARN pairs: {len(matches):,}")
    if matches.empty:
        print("No candidates passed the name/state filter.")
    else:
        print("\nCandidate tiers:")
        print(matches["candidate_tier"].value_counts().to_string())
        print("\nReview these candidate rows first:")
        cols_show = ["yelp_name", "yelp_city", "yelp_state", "warn_employer",
                     "warn_location", "warn_effective_date", "warn_notice_type",
                     "name_similarity", "candidate_tier"]
        print(matches[cols_show].head(25).to_string(index=False))
    print("\nCreated:")
    print(" data/closure_validation_sample.csv")
    print(" data/warn_candidate_matches.csv")
    print(" data/wayback_review_queue.csv")
    print("\nDo not count a candidate as a confirmed match until employer/site identity")
    print("and the event type/date are checked against the original notice.")


if __name__ == "__main__":
    main()
