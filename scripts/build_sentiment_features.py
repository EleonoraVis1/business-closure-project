"""Stream valid restaurant reviews; retain only business-level VADER aggregates."""
import argparse
import gzip
import hashlib
import json
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
import numpy as np
import pandas as pd

FIELDS = ['sentiment_mean_2021', 'sentiment_std_2021', 'negative_review_share_2021',
          'sentiment_review_count_2021', 'sentiment_mean_first_6m_2021',
          'sentiment_mean_recent_6m_2021', 'sentiment_change_recent_vs_first_6m']

def aggregate(reviews, features, analyzer):
    if features.business_id.isna().any() or features.business_id.duplicated().any():
        raise ValueError('Missing or duplicate business IDs')
    ids = set(features.business_id)
    # count, sum, sum squares, negative count, first count/sum, recent count/sum
    aggregates = {bid: np.zeros(8) for bid in ids}
    seen_businesses = set()
    total = scored = 0
    with gzip.open(reviews, 'rt', encoding='utf-8') as source:
        for line in source:
            r = json.loads(line)
            bid = r['business_id']
            if bid not in ids or not r.get('review_id'):
                raise ValueError('Unknown business ID or missing review ID')
            date = datetime.fromisoformat(r['date'])
            if not isinstance(r.get('text'), str) or not r['text'].strip():
                raise ValueError('Missing text')
            if not 1 <= r['stars'] <= 5:
                raise ValueError('Invalid review stars')
            total += 1
            seen_businesses.add(bid)
            if date.year != 2021:
                continue
            score = analyzer.polarity_scores(r['text'])['compound']
            if not np.isfinite(score) or not -1 <= score <= 1:
                raise ValueError('Invalid sentiment score')
            a = aggregates[bid]
            a[:4] += [1, score, score * score, score <= -.05]
            offset = 4 if date.month <= 6 else 6
            a[offset:offset+2] += [1, score]
            scored += 1
            if scored % 50000 == 0:
                print(f'Scored {scored:,}; streamed {total:,}', flush=True)
    rows = []
    for bid in features.business_id:
        n, s, ss, neg, nf, sf, nr, sr = aggregates[bid]
        first = sf / nf if nf else np.nan
        recent = sr / nr if nr else np.nan
        # Sample standard deviation: undefined for fewer than two reviews.
        rows.append([bid, s/n if n else np.nan,
                     np.sqrt(max(0, (ss-s*s/n)/(n-1))) if n > 1 else np.nan,
                     neg/n if n else np.nan, int(n), first, recent, recent-first])
    sentiment = pd.DataFrame(rows, columns=['business_id', *FIELDS])
    joined = features.merge(sentiment, on='business_id', how='left', validate='one_to_one', sort=False)
    if not joined.sentiment_review_count_2021.eq(joined.reviews_2021).all():
        raise ValueError('2021 count reconciliation failed')
    for field, index in [('reviews_first_6m_2021', 4), ('reviews_recent_6m_2021', 6)]:
        if not features[field].eq(features.business_id.map(lambda b: aggregates[b][index])).all():
            raise ValueError('Half-year count reconciliation failed')
    for c in features:
        pd.testing.assert_series_equal(joined[c], features[c])
    return sentiment, joined, dict(valid_reviews=total, scored_reviews_2021=scored,
        unique_review_business_ids=len(seen_businesses), restaurants=len(features),
        mismatched_counts=0, missing_sentiment_means=int(sentiment.sentiment_mean_2021.isna().sum()),
        singleton_std_missing=int(sentiment.sentiment_review_count_2021.eq(1).sum()))

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reviews', type=Path, default=Path('data/reviews.jsonl.gz'))
    p.add_argument('--features', type=Path, default=Path('data/business_features_stage1_final.csv'))
    p.add_argument('--output', type=Path, default=Path('data/business_features_with_sentiment.csv'))
    args = p.parse_args()
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    sentiment, joined, audit = aggregate(args.reviews, pd.read_csv(args.features), SentimentIntensityAnalyzer())
    sentiment.to_csv('data/sentiment_features_2021.csv', index=False)
    joined.to_csv(args.output, index=False)
    with args.reviews.open('rb') as compressed:
        review_hash = hashlib.file_digest(compressed, 'sha256').hexdigest()
    audit.update(review_file_sha256=review_hash, vader_version=version('vaderSentiment'), negative_threshold='compound <= -0.05',
                 std='sample ddof=1; NaN for n<2', review_file_bytes=args.reviews.stat().st_size,
                 stage1_sha256=hashlib.sha256(args.features.read_bytes()).hexdigest())
    out = Path('research/status_experiments'); out.mkdir(parents=True, exist_ok=True)
    (out/'sentiment_validation.json').write_text(json.dumps(audit, indent=2)+'\n')
    print(audit, flush=True)

if __name__ == '__main__': main()
