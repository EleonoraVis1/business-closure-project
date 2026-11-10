# Snapshot status feature ablation

This experiment answers the latest EDA's feature-group question using the audited
`data/business_features_stage1_final.csv`. It adds a separate experiment and does
not replace the existing baseline or research on closure dates.

## Reproduce

From the repository root, install `requirements-status-ablation.txt` in a virtual
environment, then run:

```sh
python scripts/status_feature_ablation.py
python -m unittest discover -s tests
```

Use `--input PATH --output-dir PATH` to use other locations. Outputs include model
metrics, per-business test probabilities, split membership, input SHA-256,
feature lists and runtime versions. IDs are sorted before the seeded, stratified
80/20 split, making it stable under CSV row reordering. No parameters or feature
groups were selected by optimizing test scores. Median imputation, missingness
indicators and scaling are fitted only on training data. Logistic regressions use
balanced class weights and a fixed 0.5 threshold. The dummy predicts the training
majority class. All models use exactly the same holdout businesses.

## Results

52,268 restaurants; 41,814 training rows and 10,454 test rows. Test closed share:
33.06%. Positive class is **snapshot closed**, reversing the original baseline's
open-positive probability convention.

| Features | Balanced accuracy | Closed F1 | ROC-AUC | Closed average precision |
|---|---:|---:|---:|---:|
| Majority dummy | .5000 | .0000 | .5000 | .3306 |
| Snapshot stars, review total, lifetime check-ins | .5742 | .5112 | .6112 | .4023 |
| 2021 review/tip counts | .8428 | .7621 | .9167 | .7699 |
| 2021 counts plus rating summaries | .8977 | .8442 | .9202 | .7726 |
| All groups | .9016 | .8473 | .9388 | .8534 |

2021 activity carries much more snapshot-status discrimination than lifetime
summary features alone. Adding rating summaries yields a small AUC increase but
a larger change at the fixed classification threshold; this is not evidence of
a comparably large ranking improvement. Missing rating averages also expose
activity coverage through the imputer's indicators. Adding lifetime summaries to
the 2021 groups improves AUC and average precision on this split.

These are single-holdout descriptive comparisons, without uncertainty intervals
or significance tests. Strongly overlapping review-count features remain in the
experiment to compare existing groups rather than select an optimal subset.
Random splitting does not assess geographic or temporal transfer. Neither 2021
features nor lifetime totals are established as preceding closure; inactivity
may follow closure. Removing lifetime summaries therefore does **not** remove
all timing leakage or establish early warning. The target is the Yelp snapshot,
not independently dated future closure or financial failure.

Next: if the group continues snapshot classification, predefine repeated-split
or cross-validation comparisons and a geographic holdout before tuning. For
future closure prediction, first establish cutoff operation and independently
verified outcomes, including controls; this experiment does not solve that gap.
