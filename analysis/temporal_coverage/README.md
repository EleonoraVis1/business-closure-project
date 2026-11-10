# Reproducible temporal coverage diagnostics

Run from the repository root with Python 3.9 or later; no third-party packages are needed:

```sh
python3 scripts/temporal_coverage_diagnostics.py
python3 -m unittest discover -s tests -v
```

The command reads `data/business_features.csv` and `analysis/eda_business_timeline.csv` without modifying them. It writes `summary.json` and a sorted, per-business `business_diagnostics.csv` here. The large per-business output is ignored by Git and can be regenerated. The committed summary includes SHA-256 hashes of both inputs so results can be associated with exact file versions. No raw Yelp archive or network request is needed.

Paths can be overridden with `--features`, `--timeline`, and `--output`. `--start` and `--end` accept inclusive ISO dates and change the last-review coverage window only. Count-consistency checks continue to refer to the explicitly named 2021 and October–December 2021 features regardless of coverage-window arguments.

## Results on the supplied inputs

All 52,268 feature IDs match the timeline, with no additional timeline IDs. IDs are required to be unique and nonblank; malformed input schemas or duplicate IDs stop the run rather than silently changing the denominator.

| Last observed review | Snapshot closed (`0`) | Snapshot open (`1`) |
| --- | ---: | ---: |
| Before 2021 | 15,987 (92.5%) | 4,458 (12.7%) |
| During 2021 | 1,247 (7.2%) | 20,035 (57.3%) |
| After 2021 | 47 (0.3%) | 10,494 (30.0%) |
| Total | 17,281 | 34,987 |

32,018 businesses (61.3%) have different snapshot review totals and supplied-review totals. This flag describes disagreement; it does not establish which count is wrong or why they differ. No other implemented consistency flags fired on these inputs.

The audit also checks missing timeline rows, invalid/missing dates, status disagreement, invalid or negative review counts, recent counts exceeding annual counts, and positive 2021/recent counts inconsistent with the final observed review date. Missing dates remain unknown. Issue counts overlap and are not a count of unique affected businesses. Rows with invalid status retain their original value and appear separately in the summary.

## Research interpretation and next step

Last review dates describe observed review activity, not permanent closure dates. Snapshot `is_open` provides no event time. Review silence does not establish closure, and later reviews do not prove contemporaneous operation. This audit cannot adjudicate the source records or certify eligibility for supervised future-closure prediction. It complements the existing exploratory temporal audit with machine-readable diagnostics and regression tests.

The strong difference in pre-2021 coverage suggests that a snapshot classifier using 2021 activity may partly distinguish businesses whose observed activity already ended before its feature window. Evaluate activity-coverage sensitivity descriptively before interpreting classification performance. The Philadelphia pilot reported no fully eligible closure/control records, so this contribution deliberately creates no event labels. A forward prediction design still requires independently validated cutoff operation, dated outcomes, and control follow-up.

Tests cover inclusive boundaries, unknown dates, missing and extra IDs, contradictory counts/status, duplicate and blank IDs, malformed schemas, stable ordering, repeatable output, input preservation, and reversed window rejection. These checks validate the diagnostic implementation; they do not validate external closure evidence.
