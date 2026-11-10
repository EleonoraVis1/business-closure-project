"""Compare feature groups for snapshot status; never infer future closure."""
import argparse
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SNAPSHOT = ['business_stars', 'business_review_count', 'all_time_checkins']
ACTIVITY = ['reviews_2021', 'reviews_first_6m_2021', 'reviews_recent_6m_2021',
            'reviews_recent_3m_2021', 'tips_2021', 'distinct_tip_users_2021']
RATINGS = ['avg_review_stars_2021', 'avg_review_stars_first_6m',
           'avg_review_stars_recent_6m', 'rating_change_recent_vs_first_6m']
GROUPS = {'snapshot_totals': SNAPSHOT, 'activity_2021': ACTIVITY,
          'activity_and_ratings_2021': ACTIVITY + RATINGS,
          'all_features': SNAPSHOT + ACTIVITY + RATINGS}


def validate(df):
    required = ['business_id', 'is_open'] + GROUPS['all_features']
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')
    if df.business_id.isna().any() or df.business_id.duplicated().any():
        raise ValueError('Business IDs must be present and unique')
    if not df.is_open.isin([0, 1]).all() or df.is_open.nunique() != 2:
        raise ValueError('is_open must contain both binary classes without missing labels')
    values = df[GROUPS['all_features']].apply(pd.to_numeric, errors='raise')
    if np.isinf(values.to_numpy(dtype=float)).any():
        raise ValueError('Infinite feature values')
    if (values[SNAPSHOT[1:] + ACTIVITY] < 0).any().any():
        raise ValueError('Negative count features')
    return df.assign(**{col: values[col] for col in values})


def run(input_path, output_dir):
    df = validate(pd.read_csv(input_path)).sort_values('business_id').reset_index(drop=True)
    # Sorting makes splits invariant to input row order. One held-out set for every model.
    train, test = train_test_split(df.index, stratify=df.is_open, test_size=.2, random_state=42)
    y_train, y_test = 1 - df.loc[train, 'is_open'], 1 - df.loc[test, 'is_open']
    results, predictions = [], []
    for name, columns in {'majority_dummy': ACTIVITY, **GROUPS}.items():
        if df.loc[train, columns].isna().all().any():
            raise ValueError(f'Entirely missing training feature in {name}')
        model = (DummyClassifier(strategy='most_frequent') if name == 'majority_dummy' else
                 make_pipeline(SimpleImputer(strategy='median', add_indicator=True),
                               StandardScaler(), LogisticRegression(class_weight='balanced', max_iter=2000, random_state=42)))
        model.fit(df.loc[train, columns], y_train)
        probability = model.predict_proba(df.loc[test, columns])[:, list(model.classes_).index(1)]
        predicted = model.predict(df.loc[test, columns])
        results.append(dict(model=name, balanced_accuracy=balanced_accuracy_score(y_test, predicted),
                            f1_closed=f1_score(y_test, predicted, zero_division=0),
                            roc_auc_closed=roc_auc_score(y_test, probability),
                            average_precision_closed=average_precision_score(y_test, probability)))
        predictions.append(pd.DataFrame({'business_id': df.loc[test, 'business_id'], 'model': name,
                                         'actual_closed': y_test, 'predicted_closed': predicted,
                                         'probability_closed': probability}))
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(output_dir / 'metrics.csv', index=False)
    pd.concat(predictions).to_csv(output_dir / 'predictions.csv', index=False)
    df[['business_id']].assign(split=np.where(df.index.isin(test), 'test', 'train')).to_csv(output_dir / 'split.csv', index=False)
    metadata = dict(input_sha256=hashlib.sha256(input_path.read_bytes()).hexdigest(),
                    python=platform.python_version(), pandas=pd.__version__, sklearn=sklearn.__version__,
                    random_state=42, train_rows=len(train), test_rows=len(test),
                    test_closed_prevalence=float(y_test.mean()), feature_groups=GROUPS,
                    target='1 = snapshot closed; no closure dates or future outcomes',
                    limitations='Random holdout; 2021 activity may follow closure. No early-warning or causal interpretation. No tuning on test data.')
    (output_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return pd.DataFrame(results)


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=root / 'data/business_features_stage1_final.csv')
    parser.add_argument('--output-dir', type=Path, default=root / 'analysis/status_feature_ablation')
    args = parser.parse_args()
    print(run(args.input, args.output_dir).round(4).to_string(index=False))
