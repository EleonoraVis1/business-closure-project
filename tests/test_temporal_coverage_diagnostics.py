import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from temporal_coverage_diagnostics import audit, load, run


class CoverageTests(unittest.TestCase):
    def test_invalid_counts_and_status_remain_visible(self):
        features = {'a': dict(is_open='', reviews_2021='-1',
                    reviews_recent_3m_2021='NaN', business_review_count='')}
        timeline = {'a': dict(is_open='', review_count='2', last_review_date='')}
        row = audit(features, timeline, date(2021, 1, 1), date(2021, 12, 31))[0]
        self.assertEqual(row['coverage_bucket'], 'unknown_date')
        for issue in ['invalid_snapshot_status', 'invalid_review_counts',
                      'invalid_total_review_counts', 'missing_or_invalid_last_review_date']:
            self.assertIn(issue, row['issues'])

    def test_boundaries_missing_dates_and_inconsistencies(self):
        dates = ['2020-12-31', '2021-01-01', '2021-12-31', '2022-01-01', 'bad']
        features = {str(i): dict(is_open='0', reviews_2021='2',
                    reviews_recent_3m_2021='3', business_review_count='10') for i in range(6)}
        timeline = {str(i): dict(is_open='1', review_count='11', last_review_date=d)
                    for i, d in enumerate(dates)}
        rows = audit(features, timeline, date(2021, 1, 1), date(2021, 12, 31))
        self.assertEqual([r['coverage_bucket'] for r in rows],
                         ['before_window', 'in_window', 'in_window', 'after_window',
                          'unknown_date', 'missing_timeline'])
        self.assertIn('2021_count_after_last_observation', rows[0]['issues'])
        self.assertIn('snapshot_review_count_disagreement', rows[0]['issues'])
        self.assertIn('recent_count_exceeds_annual', rows[0]['issues'])
        self.assertIn('snapshot_status_disagreement', rows[0]['issues'])
        self.assertNotIn('2021_count_after_last_observation', rows[1]['issues'])
        self.assertEqual(rows, audit(dict(reversed(list(features.items()))), timeline,
                                    date(2021, 1, 1), date(2021, 12, 31)))

    def test_duplicate_blank_and_schema_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'input.csv'
            for text in ['business_id\na\na\n', 'business_id\n""\n', 'other\na\n']:
                path.write_text(text)
                with self.assertRaises(ValueError):
                    load(path, ['business_id'])

    def test_cli_outputs_reproduce_and_preserve_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            features, timeline = root / 'features.csv', root / 'timeline.csv'
            features.write_text('business_id,is_open,business_review_count,reviews_2021,reviews_recent_3m_2021\na,1,2,0,0\n')
            timeline.write_text('business_id,is_open,review_count,last_review_date\na,1,2,2022-01-01\nb,0,1,2020-01-01\n')
            before = features.read_bytes(), timeline.read_bytes()
            args = features, timeline, root / 'out', date(2021, 1, 1), date(2021, 12, 31)
            first = run(*args)
            result = (root / 'out/business_diagnostics.csv').read_bytes()
            self.assertEqual(first, run(*args))
            self.assertEqual(result, (root / 'out/business_diagnostics.csv').read_bytes())
            self.assertEqual(first['timeline_ids_without_features'], 1)
            self.assertEqual(before, (features.read_bytes(), timeline.read_bytes()))
            with self.assertRaises(ValueError):
                audit({}, {}, date(2022, 1, 1), date(2021, 1, 1))


if __name__ == '__main__':
    unittest.main()
