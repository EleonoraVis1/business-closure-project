
import argparse
import re
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DEFAULT_INPUT = DATA / "wayback_review_queue.csv"
SITES_OUT = DATA / "wayback_discovered_sites.csv"
CAPTURES_OUT = DATA / "wayback_capture_summary.csv"
REVIEW_OUT = DATA / "wayback_manual_review.csv"

CDX = "https://web.archive.org/cdx/search/cdx"
BING = "https://www.bing.com/search"
PATHS = ["/", "/about", "/about-us", "/news", "/contact", "/locations", "/our-story", "/press", "/blog"]
EXCLUDED = (
    "yelp.com", "facebook.com", "instagram.com", "tripadvisor.com",
    "doordash.com", "ubereats.com", "grubhub.com", "google.com", "bing.com",
    "yellowpages.com", "mapquest.com", "opentable.com", "toasttab.com",
)
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/130 Safari/537.36"}


def norm(value):
    if pd.isna(value):
        return ""
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(value).lower()).split())


def domain(url):
    host = (urlparse(url).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def canonical(url):
    url = str(url or "").strip()
    if not url or url.lower() == "nan":
        return ""
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    p = urlparse(url)
    return f"https://{p.netloc}/" if p.hostname else ""


def excluded(url):
    host = (urlparse(url).hostname or "").lower()
    return any(host == x or host.endswith("." + x) for x in EXCLUDED)


def score_result(name, city, state, address, title, snippet, url):
    n = set(norm(name).split())
    if not n:
        return 0
    title_t, snippet_t = set(norm(title).split()), set(norm(snippet).split())
    domain_t = set(norm(domain(url).replace(".", " ")).split())
    city_t = set(norm(city).split())
    state_t = set(norm(state).split())
    address_t = set(norm(address).split())
    overlap = lambda x: len(n & x) / len(n)
    score = (
        .48 * overlap(title_t) + .22 * overlap(snippet_t)
        + .18 * overlap(domain_t)
        + .06 * float(bool(city_t) and city_t.issubset(title_t | snippet_t))
        + .03 * float(bool(state_t) and state_t.issubset(title_t | snippet_t))
        + .03 * (len(address_t & (title_t | snippet_t)) / len(address_t) if address_t else 0)
    )
    return round(min(score, 1), 3)


def search_bing(session, query, limit=8):
    r = session.get(BING, params={"q": query, "count": limit}, timeout=25)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    results = []
    for li in soup.select("li.b_algo"):
        a = li.select_one("h2 a")
        if not a:
            continue
        url = a.get("href", "")
        if not url.startswith("http") or excluded(url):
            continue
        p = li.select_one(".b_caption p") or li.select_one("p")
        results.append({
            "url": canonical(url), "title": a.get_text(" ", strip=True),
            "snippet": p.get_text(" ", strip=True) if p else "",
        })
        if len(results) >= limit:
            break
    return results


def discover(session, row, limit):
    name = str(row.get("name", row.get("yelp_name", ""))).strip()
    city = str(row.get("city", row.get("yelp_city", ""))).strip()
    state = str(row.get("state", row.get("yelp_state", ""))).strip()
    address = str(row.get("address", "")).strip()
    found = []
    supplied = canonical(row.get("official_url", ""))
    if supplied and not excluded(supplied):
        found.append({"url": supplied, "title": "Input official_url", "snippet": "", "source": "input_csv", "score": 1.0})
    queries = [
        f'"{name}" "{city}" "{state}" official website restaurant',
        f'"{name}" "{city}" restaurant {address}' if address else f'"{name}" "{city}" restaurant',
    ]
    seen = {domain(x["url"]) for x in found}
    for q in queries:
        try:
            results = search_bing(session, q, limit)
        except Exception as e:
            print(f"  Search warning: {e}")
            continue
        for item in results:
            d = domain(item["url"])
            if not d or d in seen or excluded(item["url"]):
                continue
            seen.add(d)
            item.update(source="bing_html", score=score_result(
                name, city, state, address, item["title"], item["snippet"], item["url"]
            ))
            found.append(item)
        time.sleep(1)
    found.sort(key=lambda x: x.get("score", 0), reverse=True)
    return found[:limit]


def query_cdx(session, url):
    params = {
        "url": url, "output": "json",
        "fl": "timestamp,original,statuscode,mimetype,digest",
        "filter": "statuscode:200", "collapse": "timestamp:6",
    }
    r = session.get(CDX, params=params, timeout=35)
    if r.status_code == 429:
        time.sleep(8)
        r = session.get(CDX, params=params, timeout=35)
    r.raise_for_status()
    data = r.json()
    if not data or len(data) < 2:
        return []
    header = data[0]
    rows = []
    for values in data[1:]:
        if not isinstance(values, list):
            continue
        item = dict(zip(header, values))
        ts = item.get("timestamp", "")
        try:
            dt = datetime.strptime(ts[:14], "%Y%m%d%H%M%S")
            item["captured_at"] = dt.isoformat(sep=" ")
        except ValueError:
            item["captured_at"] = ""
        item["archive_url"] = f"https://web.archive.org/web/{ts}/{item.get('original', url)}" if ts else ""
        rows.append(item)
    return sorted(rows, key=lambda x: x.get("timestamp", ""))


def target_date(row):
    for key in ("suspected_closure_date", "closure_date", "first_confirmed_closed",
                "estimated_closure_date", "last_review_date"):
        value = row.get(key, "")
        if str(value).strip() and str(value).lower() != "nan":
            parsed = pd.to_datetime(value, errors="coerce")
            if pd.notna(parsed):
                return parsed.to_pydatetime()
    return None


def nearest(captures, target):
    if not captures or target is None:
        return None
    valid = []
    for c in captures:
        dt = pd.to_datetime(c.get("captured_at", ""), errors="coerce")
        if pd.notna(dt):
            valid.append((abs((dt.to_pydatetime() - target).total_seconds()), c))
    return min(valid, key=lambda x: x[0])[1] if valid else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(DEFAULT_INPUT))
    ap.add_argument("--max-businesses", type=int, default=0, help="0 means all")
    ap.add_argument("--max-sites-per-business", type=int, default=3)
    ap.add_argument("--search-results", type=int, default=8)
    ap.add_argument("--delay", type=float, default=1.0)
    args = ap.parse_args()
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = ROOT / input_path
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input CSV: {input_path}")
    df = pd.read_csv(input_path).fillna("")
    for alternatives, label in [
        (("name", "yelp_name"), "name"),
        (("city", "yelp_city"), "city"),
        (("state", "yelp_state"), "state"),
    ]:
        if not any(c in df.columns for c in alternatives):
            raise ValueError(f"Input CSV needs a {label} column ({alternatives}).")
    if args.max_businesses:
        df = df.head(args.max_businesses)
    DATA.mkdir(exist_ok=True)
    session = requests.Session()
    session.headers.update(HEADERS)
    site_rows, capture_rows = [], []

    for i, (_, row) in enumerate(df.iterrows(), 1):
        name = str(row.get("name", row.get("yelp_name", "")))
        city = str(row.get("city", row.get("yelp_city", "")))
        state = str(row.get("state", row.get("yelp_state", "")))
        bid = str(row.get("business_id", ""))
        print(f"[{i}/{len(df)}] {name} — {city}, {state}")
        candidates = discover(session, row, args.search_results)
        if not candidates:
            site_rows.append({
                "business_id": bid, "business_name": name, "city": city, "state": state,
                "candidate_rank": "", "candidate_url": "", "domain": "",
                "search_score": "", "discovery_source": "no_results",
                "manual_decision": "", "notes": "No candidate domain discovered",
            })
            continue
        target = target_date(row)
        for rank, candidate in enumerate(candidates[:args.max_sites_per_business], 1):
            site = candidate["url"]
            site_rows.append({
                "business_id": bid, "business_name": name, "city": city, "state": state,
                "candidate_rank": rank, "candidate_url": site, "domain": domain(site),
                "search_score": candidate.get("score", ""),
                "discovery_source": candidate.get("source", ""),
                "search_result_title": candidate.get("title", ""),
                "search_result_snippet": candidate.get("snippet", ""),
                "manual_decision": "", "notes": "",
            })
            for path in PATHS:
                page_url = site.rstrip("/") + path
                try:
                    caps = query_cdx(session, page_url)
                    err = ""
                except Exception as e:
                    caps, err = [], str(e)
                first = caps[0] if caps else {}
                last = caps[-1] if caps else {}
                near = nearest(caps, target)
                capture_rows.append({
                    "business_id": bid, "business_name": name, "city": city, "state": state,
                    "candidate_url": site, "page_path": path, "query_url": page_url,
                    "capture_count": len(caps),
                    "earliest_capture": first.get("captured_at", ""),
                    "earliest_archive_url": first.get("archive_url", ""),
                    "latest_capture": last.get("captured_at", ""),
                    "latest_archive_url": last.get("archive_url", ""),
                    "nearest_target_capture": near.get("captured_at", "") if near else "",
                    "nearest_target_archive_url": near.get("archive_url", "") if near else "",
                    "target_date_used": target.date().isoformat() if target else "",
                    "error": err,
                })
                time.sleep(args.delay)
            # Incremental saves: can resume evidence even if a long run stops.
            pd.DataFrame(site_rows).to_csv(SITES_OUT, index=False)
            pd.DataFrame(capture_rows).to_csv(CAPTURES_OUT, index=False)

    sites = pd.DataFrame(site_rows)
    caps = pd.DataFrame(capture_rows)
    if not caps.empty:
        numeric = pd.to_numeric(caps["capture_count"], errors="coerce").fillna(0)
        caps["_has_capture"] = numeric.gt(0)
        review = caps.groupby(
            ["business_id", "business_name", "city", "state", "candidate_url"],
            dropna=False,
        ).agg(
            pages_checked=("page_path", "count"),
            pages_with_captures=("_has_capture", "sum"),
            total_captures=("capture_count", lambda s: pd.to_numeric(s, errors="coerce").fillna(0).sum()),
            earliest_capture=("earliest_capture", "min"),
            latest_capture=("latest_capture", "max"),
            errors=("error", lambda s: sum(bool(x) for x in s)),
        ).reset_index()
        review["manual_decision"] = ""
        review["closure_evidence_found"] = ""
        review["closure_date_supported_by_evidence"] = ""
        review["evidence_notes"] = ""
        review.to_csv(REVIEW_OUT, index=False)

    print("\nDone.")
    print(f"Discovered sites: {len(sites)} -> {SITES_OUT}")
    print(f"Page capture rows: {len(caps)} -> {CAPTURES_OUT}")
    print(f"Manual review summary: {REVIEW_OUT}")
    print("Remember: archived capture timestamps are not closure dates.")

if __name__ == "__main__":
    main()
