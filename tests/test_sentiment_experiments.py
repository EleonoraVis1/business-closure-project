"""Small, independent checks for sentiment aggregation and experimental design."""
import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from build_sentiment_features import aggregate
from recorded_status_experiments import EXPERIMENT_GROUPS, estimator, scores, load_and_validate

class Analyzer:
    def polarity_scores(self, text):
        return {'compound': float(text)}

class SentimentTests(unittest.TestCase):
    def fixture(self, directory, records=None):
        path = Path(directory) / 'reviews.gz'
        records = records if records is not None else [
            ('a','2020-12-31 23:59:59','-1'),
            ('a','2021-01-01 00:00:00','-0.05'),
            ('a','2021-06-30 23:59:59','0.5'),
            ('a','2021-07-01 00:00:00','1'),
            ('a','2021-12-31 23:59:59','-1'),
            ('a','2022-01-01 00:00:00','1'),
            ('b','2021-08-01 00:00:00','0.2')]
        with gzip.open(path,'wt') as stream:
            for i,(bid,date,text) in enumerate(records):
                stream.write(json.dumps(dict(business_id=bid,date=date,text=text,review_id=str(i),stars=3))+'\n')
        features = pd.DataFrame(dict(business_id=['a','b','c'],is_open=[0,1,1],
            reviews_2021=[4,1,0],reviews_first_6m_2021=[2,0,0],reviews_recent_6m_2021=[2,1,0]))
        return path, features

    def test_boundaries_aggregation_missingness_and_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            path, features = self.fixture(directory)
            sentiment, combined, audit = aggregate(path,features,Analyzer())
            a,b,c = [sentiment.iloc[i] for i in range(3)]
            values = [-.05,.5,1,-1]
            self.assertAlmostEqual(a.sentiment_mean_2021,np.mean(values))
            self.assertAlmostEqual(a.sentiment_std_2021,np.std(values,ddof=1))
            self.assertEqual(a.negative_review_share_2021,.5)
            self.assertAlmostEqual(a.sentiment_mean_first_6m_2021,.225)
            self.assertAlmostEqual(a.sentiment_change_recent_vs_first_6m,-.225)
            self.assertTrue(np.isnan(b.sentiment_std_2021))
            self.assertTrue(np.isnan(c.sentiment_mean_2021))
            self.assertEqual(c.sentiment_review_count_2021,0)
            self.assertEqual(audit['scored_reviews_2021'],5)
            pd.testing.assert_frame_equal(combined[features.columns], features)

    def test_invalid_inputs_and_reconciliation(self):
        with tempfile.TemporaryDirectory() as directory:
            for record in [('unknown','2021-01-01','0'),('a','2021-02-30','0'),('a','2021-01-01','1.1')]:
                path, features = self.fixture(directory,[record])
                with self.assertRaises(ValueError): aggregate(path,features,Analyzer())
            path, features = self.fixture(directory)
            features.loc[0,'reviews_2021']=5
            with self.assertRaises(ValueError): aggregate(path,features,Analyzer())
            features.loc[1,'business_id']='a'
            with self.assertRaises(ValueError): aggregate(path,features,Analyzer())

    def test_real_vader_bounds(self):
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        analyzer=SentimentIntensityAnalyzer()
        self.assertGreater(analyzer.polarity_scores('Wonderful delicious food!')['compound'],.05)
        self.assertLess(analyzer.polarity_scores('Disgusting terrible food!')['compound'],-.05)

    def test_groups_and_preprocessing_do_not_fit_test_data(self):
        for columns in EXPERIMENT_GROUPS.values():
            self.assertEqual(len(columns),len(set(columns)))
            self.assertNotIn('is_open',columns)
            self.assertFalse('reviews_2021' in columns and 'reviews_first_6m_2021' in columns)
        for kind in ['logistic_regression','random_forest']:
            model=estimator(kind,['business_stars'],10,42)
            model.fit(pd.DataFrame({'business_stars':[1.,2.,3.,np.nan]}),[0,1,0,1])
            pipeline=model.named_steps['preprocess'].named_transformers_['numeric']
            imputer=pipeline.named_steps['impute'] if kind=='logistic_regression' else pipeline
            before=imputer.statistics_.copy()
            model.predict_proba(pd.DataFrame({'business_stars':[10000.,np.nan]}))
            np.testing.assert_equal(before,imputer.statistics_)
            self.assertEqual(before[0],2.)
        output=scores(pd.Series([1,0]),np.array([.9,.1]),np.array([1,0]))
        self.assertEqual(output['f1_closed'],1.)

if __name__=='__main__': unittest.main()
