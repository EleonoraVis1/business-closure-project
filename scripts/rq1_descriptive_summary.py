#!/usr/bin/env python3
"""Summarize RQ1 for Yelp snapshot status. Never infer a closure date."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

CANDIDATES = ['business_stars', 'business_review_count', 'reviews_2021',
              'reviews_first_6m_2021', 'reviews_recent_6m_2021',
              'reviews_recent_3m_2021', 'avg_review_stars_2021',
              'rating_change_recent_vs_first_6m', 'all_time_checkins',
              'tips_2021', 'sentiment_mean_2021', 'sentiment_std_2021',
              'negative_review_share_2021', 'sentiment_change_recent_vs_first_6m']

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    df = pd.read_csv(args.input)
    if df.business_id.duplicated().any():
        raise ValueError('Duplicate business IDs')
    if not df.is_open.isin([0,1]).all():
        raise ValueError('Unexpected status labels')
    rows=[]
    for c in CANDIDATES:
        if c not in df.columns:
            continue
        for status in [0,1]:
            x = pd.to_numeric(df.loc[df.is_open.eq(status), c], errors='raise')
            v = x.dropna().to_numpy()
            rows.append({'feature':c, 'recorded_status': 'closed' if status==0 else 'open',
                         'businesses':len(x), 'non_missing':len(v),
                         'missing_percent':100*x.isna().mean(),
                         'zero_percent_all': 100*x.eq(0).mean(),
                         'mean_observed': float(np.mean(v)) if len(v) else np.nan,
                         'median_observed':float(np.median(v)) if len(v) else np.nan,
                         'std_observed':float(np.std(v,ddof=1)) if len(v)>1 else np.nan})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output,index=False)
    from scipy.stats import mannwhitneyu, ttest_ind
    comparisons = []
    for c in CANDIDATES:
        if c not in df: continue
        a = df.loc[df.is_open.eq(0), c].dropna().to_numpy()
        b = df.loc[df.is_open.eq(1), c].dropna().to_numpy()
        if min(len(a), len(b)) < 2: continue
        u, p = mannwhitneyu(a, b, alternative='two-sided')
        pooled = np.sqrt(((len(a)-1)*a.var(ddof=1)+(len(b)-1)*b.var(ddof=1))/(len(a)+len(b)-2))
        comparisons.append(dict(feature=c, n_closed=len(a), n_open=len(b),
            mean_difference_closed_minus_open=a.mean()-b.mean(),
            hedges_g=(1-3/(4*(len(a)+len(b))-9))*(a.mean()-b.mean())/pooled,
            rank_biserial_closed_minus_open=2*u/(len(a)*len(b))-1,
            mann_whitney_p=p, welch_p=ttest_ind(a,b,equal_var=False).pvalue))
    comparisons = pd.DataFrame(comparisons)
    # Benjamini-Hochberg adjustment within the descriptive feature family.
    pvalues = comparisons.mann_whitney_p.to_numpy()
    order = np.argsort(pvalues)
    adjusted = np.minimum.accumulate((pvalues[order]*len(order)/np.arange(1,len(order)+1))[::-1])[::-1]
    q = np.empty(len(order)); q[order] = np.minimum(adjusted,1)
    comparisons['mann_whitney_bh_q'] = q
    comparisons.to_csv(args.output.parent / 'rq1_comparisons.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,3,figsize=(12,4))
    for status, color in [(0,'#c56b42'),(1,'#287c8e')]:
        group = df.loc[df.is_open.eq(status)]
        label = 'Recorded closed' if status == 0 else 'Recorded open'
        axes[0].hist(np.log1p(group.reviews_2021),bins=np.linspace(0,np.log1p(df.reviews_2021.max()),36),density=True,histtype='step',label=label,color=color)
        axes[1].hist(group.sentiment_mean_2021.dropna(),bins=np.linspace(-1,1,36),density=True,histtype='step',label=label,color=color)
        axes[2].hist(group.negative_review_share_2021.dropna(),bins=np.linspace(0,1,26),density=True,histtype='step',label=label,color=color)
    for ax, title in zip(axes, ['log(1 + 2021 reviews), all restaurants', 'Mean sentiment, observed only', 'Negative share, observed only']):
        ax.set_title(title,fontsize=10); ax.set_ylabel('Density')
    axes[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(args.output.parent / 'rq1_distributions.png',dpi=180); plt.close(fig)
    print('Saved',args.output)

if __name__ == '__main__': main()
