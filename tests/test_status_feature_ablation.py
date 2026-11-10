"""Run with: python -m unittest discover -s tests."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from status_feature_ablation import GROUPS, run, validate


class AblationTests(unittest.TestCase):
    def fixture(self):
        rng = np.random.default_rng(7)
        df = pd.DataFrame({c: rng.integers(0, 10, 100).astype(float) for c in GROUPS['all_features']})
        df['business_id'] = [f'b{i:03}' for i in range(100)]
        df['is_open'] = [0, 1] * 50
        df.loc[::3, 'avg_review_stars_2021'] = np.nan
        return df

    def test_invalid_labels_and_duplicate_ids_rejected(self):
        for column, value in [('is_open', 3), ('business_id', 'b001')]:
            df = self.fixture()
            df.loc[0, column] = value
            with self.assertRaises(ValueError):
                validate(df)

    def test_common_split_and_row_order_reproducibility(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'features.csv'
            df = self.fixture()
            df.to_csv(source, index=False)
            run(source, root / 'first')
            df.sample(frac=1, random_state=8).to_csv(source, index=False)
            run(source, root / 'second')
            for name in ['metrics.csv', 'predictions.csv', 'split.csv']:
                pd.testing.assert_frame_equal(pd.read_csv(root / 'first' / name), pd.read_csv(root / 'second' / name))
            split = pd.read_csv(root / 'first/split.csv')
            predictions = pd.read_csv(root / 'first/predictions.csv')
            test_ids = set(split.loc[split.split == 'test', 'business_id'])
            self.assertEqual(len(test_ids), 20)
            self.assertTrue(test_ids.isdisjoint(set(split.loc[split.split == 'train', 'business_id'])))
            for _, group in predictions.groupby('model'):
                self.assertEqual(set(group.business_id), test_ids)
                self.assertTrue(group.probability_closed.between(0, 1).all())


if __name__ == '__main__':
    unittest.main()
