"""Verify cached sentiment and experiment artifacts without rescoring reviews."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research/status_experiments'

def main():
    stage=pd.read_csv(ROOT/'data/business_features_stage1_final.csv')
    full=pd.read_csv(ROOT/'data/business_features_with_sentiment.csv')
    sentiment=pd.read_csv(ROOT/'data/sentiment_features_2021.csv')
    audit=json.loads((OUT/'sentiment_validation.json').read_text())
    assert hashlib.sha256((ROOT/'data/business_features_stage1_final.csv').read_bytes()).hexdigest()==audit['stage1_sha256']
    with (ROOT/'data/reviews.jsonl.gz').open('rb') as source:
        assert hashlib.file_digest(source,'sha256').hexdigest()==audit['review_file_sha256']
    pd.testing.assert_frame_equal(full[stage.columns],stage)
    pd.testing.assert_frame_equal(full[sentiment.columns],sentiment)
    assert len(full)==52268 and full.business_id.is_unique and sentiment.business_id.is_unique
    assert full.sentiment_review_count_2021.eq(full.reviews_2021).all()
    assert full.reviews_2021.sum()==409047==audit['scored_reviews_2021']
    assert audit['valid_reviews']==4724471 and audit['unique_review_business_ids']==52268
    for c,lo,hi in [('sentiment_mean_2021',-1,1),('negative_review_share_2021',0,1),
        ('sentiment_mean_first_6m_2021',-1,1),('sentiment_mean_recent_6m_2021',-1,1),
        ('sentiment_change_recent_vs_first_6m',-2,2),('sentiment_std_2021',0,2)]:
        assert full[c].dropna().between(lo,hi).all(), c
    assert full.sentiment_mean_2021.isna().equals(full.reviews_2021.eq(0))
    assert full.negative_review_share_2021.isna().equals(full.reviews_2021.eq(0))
    assert full.sentiment_std_2021.isna().equals(full.reviews_2021.lt(2))
    assert full.sentiment_mean_first_6m_2021.isna().equals(full.reviews_first_6m_2021.eq(0))
    assert full.sentiment_mean_recent_6m_2021.isna().equals(full.reviews_recent_6m_2021.eq(0))
    np.testing.assert_allclose(full.sentiment_change_recent_vs_first_6m,
        full.sentiment_mean_recent_6m_2021-full.sentiment_mean_first_6m_2021,equal_nan=True)
    populations={}
    for population,df in [('main',full),('active',full.loc[full.reviews_2021.gt(0)])]:
        split=pd.read_csv(OUT/population/'split.csv')
        pred=pd.read_csv(OUT/population/'predictions.csv')
        metrics=pd.read_csv(OUT/population/'model_comparison.csv')
        assert split.business_id.is_unique and set(split.business_id)==set(df.business_id)
        train=set(split.loc[split.split.eq('train'),'business_id'])
        test=set(split.loc[split.split.eq('test'),'business_id'])
        assert train.isdisjoint(test) and len(train)+len(test)==len(df)
        true_labels=(1-df.set_index('business_id').is_open).to_dict()
        for (model,group),p in pred.groupby(['model','feature_group']):
            assert p.business_id.is_unique and set(p.business_id)==test
            assert p.probability_closed.between(0,1).all()
            assert p.actual_closed.eq(p.business_id.map(true_labels)).all()
            row=metrics.loc[metrics.model.eq(model)&metrics.feature_group.eq(group)].iloc[0]
            matrix=confusion_matrix(p.actual_closed,p.predicted_closed,labels=[0,1]).ravel()
            np.testing.assert_array_equal(matrix,row[['tn_open','fp_closed','fn_closed','tp_closed']].to_numpy())
        populations[population]=dict(n=len(df),closed=int(df.is_open.eq(0).sum()),
            open=int(df.is_open.eq(1).sum()),train=len(train),test=len(test),
            models=len(metrics),common_holdout_verified=True)
    results=dict(all_checks_passed=True,stage1_unchanged=True,counts_reconciled=True,
                 bounds_and_missingness_verified=True,populations=populations)
    (OUT/'validation_checks.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))

if __name__=='__main__': main()
