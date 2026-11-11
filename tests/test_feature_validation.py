"""Smoke checks for the feature table and split-independent audit logic.

Run from repository root with: python -m unittest discover -s tests
"""
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from recorded_status_experiments import load_and_validate, estimator, scores

class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            'business_id': ['a', 'b', 'c', 'd'], 'is_open': [0, 1, 0, 1],
            'business_stars': [2.5, 4, 3, 4.5],
            'business_review_count': [10, 20, 30, 40],
            'reviews_2021': [0, 2, 1, 3],
            'reviews_first_6m_2021': [0, 1, 1, 2],
            'reviews_recent_6m_2021': [0, 1, 0, 1],
            'reviews_recent_3m_2021': [0, 1, 0, 1],
            'avg_review_stars_2021': [np.nan, 4, 3, 4.5],
            'rating_change_recent_vs_first_6m': [np.nan, 0, np.nan, 0.5],
            'all_time_checkins': [1, 10, 5, 20], 'tips_2021': [0, 1, 0, 2]
        })

    def test_valid_table(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as d:
            p = Path(d) / 'features.csv'
            self.df.to_csv(p, index=False)
            self.assertEqual(len(load_and_validate(p)), 4)

    def test_non_reconciling_reviews_fail(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as d:
            p = Path(d) / 'features.csv'
            bad = self.df.copy()
            bad.loc[0, 'reviews_2021'] = 2
            bad.to_csv(p, index=False)
            with self.assertRaises(ValueError):
                load_and_validate(p)

    def test_closed_positive_metrics(self):
        y = pd.Series([1, 0, 1, 0])
        prob = np.array([0.9, 0.1, 0.8, 0.2])
        output = scores(y, prob, (prob >= .5).astype(int))
        self.assertAlmostEqual(output['roc_auc_closed'], 1.0)
        self.assertAlmostEqual(output['f1_closed'], 1.0)

    def test_model_preprocessing(self):
        columns = ['business_stars', 'reviews_2021']
        y = pd.Series([1, 0, 1, 0])
        for kind in ('logistic_regression', 'random_forest'):
            model = estimator(kind, columns, 10, 42)
            model.fit(self.df[columns], y)
            self.assertEqual(model.predict_proba(self.df[columns]).shape, (4, 2))

if __name__ == '__main__':
    unittest.main()
