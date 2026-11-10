"""Audit existing EDA outputs without deriving closure labels (standard library only)."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import date
from pathlib import Path


def load(path, required):
    with path.open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        missing = set(required) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f'{path}: missing columns {sorted(missing)}')
        rows = {}
        for row in reader:
            key = row['business_id'].strip()
            if not key or key in rows:
                raise ValueError(f'{path}: blank or duplicate business_id {key!r}')
            rows[key] = row
        return rows


def count(value):
    try:
        number = int(value)
        return number if number >= 0 else None
    except (ValueError, TypeError):
        return None


def audit(features, timeline, start, end):
    if start > end:
        raise ValueError('Window start must be on or before end')
    results = []
    for key, feature in sorted(features.items()):
        observed = timeline.get(key)
        issues = []
        status = feature['is_open']
        if status not in ('0', '1'):
            issues.append('invalid_snapshot_status')
        annual = count(feature['reviews_2021'])
        recent = count(feature['reviews_recent_3m_2021'])
        if annual is None or recent is None:
            issues.append('invalid_review_counts')
        elif recent > annual:
            issues.append('recent_count_exceeds_annual')
        bucket = 'missing_timeline'
        last = None
        if observed is None:
            issues.append('missing_timeline')
        else:
            if observed['is_open'] != status:
                issues.append('snapshot_status_disagreement')
            try:
                last = date.fromisoformat(observed['last_review_date'])
            except (ValueError, TypeError):
                issues.append('missing_or_invalid_last_review_date')
            bucket = ('unknown_date' if last is None else
                      'before_window' if last < start else
                      'in_window' if last <= end else 'after_window')
            if last and annual is not None and annual > 0 and last < date(2021, 1, 1):
                issues.append('2021_count_after_last_observation')
            if last and recent is not None and recent > 0 and last < date(2021, 10, 1):
                issues.append('recent_count_after_last_observation')
            snapshot_count = count(feature['business_review_count'])
            observed_count = count(observed['review_count'])
            if snapshot_count is None or observed_count is None:
                issues.append('invalid_total_review_counts')
            elif snapshot_count != observed_count:
                issues.append('snapshot_review_count_disagreement')
        results.append(dict(business_id=key, snapshot_status=status,
                            last_review_date=last.isoformat() if last else '',
                            coverage_bucket=bucket, issues=';'.join(issues)))
    return results


def run(feature_path, timeline_path, output, start, end):
    features = load(feature_path, ['business_id', 'is_open', 'business_review_count',
                                   'reviews_2021', 'reviews_recent_3m_2021'])
    timeline = load(timeline_path, ['business_id', 'is_open', 'review_count', 'last_review_date'])
    rows = audit(features, timeline, start, end)
    summary = {
        'window_start': start.isoformat(), 'window_end': end.isoformat(),
        'feature_businesses': len(features), 'timeline_businesses': len(timeline),
        'timeline_ids_without_features': len(timeline.keys() - features.keys()),
        'inputs': {name: {'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                   for name, path in [('features', feature_path), ('timeline', timeline_path)]},
        'coverage_by_snapshot_status': {
            status: dict(sorted(Counter(r['coverage_bucket'] for r in rows
                                        if r['snapshot_status'] == status).items()))
            for status in sorted({r['snapshot_status'] for r in rows})},
        'issue_counts': dict(sorted(Counter(issue for r in rows
                                            for issue in r['issues'].split(';') if issue).items())),
        'interpretation': 'Last review is an activity observation, not a closure date. '
                          'Snapshot status and activity gaps do not establish prospective outcomes.',
    }
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'business_diagnostics.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['business_id', 'snapshot_status',
                               'last_review_date', 'coverage_bucket', 'issues'])
        writer.writeheader()
        writer.writerows(rows)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    return summary


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', type=Path, default=root / 'data/business_features.csv')
    parser.add_argument('--timeline', type=Path, default=root / 'analysis/eda_business_timeline.csv')
    parser.add_argument('--output', type=Path, default=root / 'analysis/temporal_coverage')
    parser.add_argument('--start', type=date.fromisoformat, default=date(2021, 1, 1))
    parser.add_argument('--end', type=date.fromisoformat, default=date(2021, 12, 31))
    args = parser.parse_args()
    print(json.dumps(run(args.features, args.timeline, args.output, args.start, args.end), indent=2))
