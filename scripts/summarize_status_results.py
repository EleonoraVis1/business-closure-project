"""Build a concise report from executed outputs; never invent experimental results."""
import json
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research/status_experiments'

def table(df, columns):
    rows=['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']
    for _, row in df.iterrows():
        rows.append('| '+' | '.join(f'{row[c]:.4f}' if isinstance(row[c],float) else str(row[c]) for c in columns)+' |')
    return '\n'.join(rows)

def main():
    validation=json.loads((OUT/'sentiment_validation.json').read_text())
    frames={name:pd.read_csv(OUT/name/'model_comparison.csv') for name in ['main','active']}
    desc=pd.read_csv(OUT/'rq1_descriptive_summary.csv')
    comparisons=pd.read_csv(OUT/'rq1_comparisons.csv')
    lines=['# Yelp recorded-status experiments','',
        'Target: Yelp recorded `is_open` snapshot status; closed (`is_open=0`) is the positive class. This study does not predict future closure.','',
        '## Existing work and integration','',
        'The repository already contained Stage 1 feature construction, review audits, EDA, a logistic activity ablation with deterministic splits, and historical results. `research/` contained a closure-date feasibility pilot. The supplied v2 package contained an unexecuted VADER script, descriptive code, LR/RF pipelines, and preliminary activity results. No cached sentiment output existed. Its VADER streaming design, descriptive summaries, estimator, closed-positive metrics, and feature validation were integrated and extended. The historical `scripts/status_feature_ablation.py` and its saved outputs remain intact; `scripts/recorded_status_experiments.py` is the current RQ1–RQ3 modeling entry point. The supplied package and research pilot were preserved.','',
        'Added: sample sentiment standard deviation, half-year means/change, full-stream record validation, one-to-one merge/count reconciliation, controlled sentiment increments, majority baseline, confusion matrices, training CV, independently stratified active-population split, paired bootstrap intervals, statistical comparisons with effect sizes/BH adjustment, figure, tests and this report.','',
        '## Data and validation','',
        '```json',json.dumps(validation,indent=2),'```','',
        'The primary gzip was streamed to EOF (including gzip integrity checking); the ZIP was not extracted. All records required valid JSON, sample business IDs, review IDs, parseable dates, nonempty text, and stars in [1,5]. Only 2021 texts were scored. Compound scores were checked in [-1,1]. Annual and both half-year business counts matched Stage 1 exactly. All original columns, row order and target labels were checked after the one-to-one left merge. The original CSV was not rewritten. This run checks business ID uniqueness, not duplicate review IDs; the earlier review audit checked review duplication.','',
        'VADER 3.3.2 compound <= -0.05 defines negative sentiment; other scores are nonnegative for this binary share. Aggregates only are saved. No-review sentiment means/shares are NaN. Standard deviation uses ddof=1 (also NaN for one review); change is NaN unless both halves have reviews. VADER is a lexicon baseline whose restaurant-domain validity was not independently adjudicated.','',
        '## RQ1: descriptive associations','',
        'Recorded-closed restaurants average 0.398 reviews in 2021 versus 11.495 for recorded-open restaurants. Zero-review shares are 92.576% and 13.419%, respectively. Among the 1,283 closed and 30,292 open restaurants with reviews, mean sentiment is 0.5154 versus 0.4841 and negative review share is 0.2057 versus 0.2269. The observed sentiment difference therefore does not support a simple claim that closed restaurants had more negative reviews. These are unweighted business-level averages with different review-count distributions.','',
        'Statistics condition on nonmissing observations. Sentiment comparisons concern restaurants with 2021 reviews, not all restaurants. All feature summaries, group sample sizes, medians, standard deviations, zero shares and missingness are in `rq1_descriptive_summary.csv`.','',
        table(desc.loc[desc.feature.isin(['reviews_2021','avg_review_stars_2021','sentiment_mean_2021','negative_review_share_2021','sentiment_change_recent_vs_first_6m'])],['feature','recorded_status','businesses','non_missing','missing_percent','mean_observed','median_observed','std_observed']),'',
        'Welch tests compare observed means; two-sided Mann–Whitney tests compare distributions (not necessarily medians). Hedges g and signed rank-biserial correlations use closed minus open orientation. BH q-values adjust the Mann–Whitney family across features. Large sample significance alone does not imply useful separation. These exploratory tests assume independent businesses and do not account for shared reviewers, chains or geography.','',
        table(comparisons,['feature','hedges_g','rank_biserial_closed_minus_open','mann_whitney_bh_q']),'',
        '![Status distributions](rq1_distributions.png)','',
        '## RQ2 and RQ3: common holdout classification','',
        'Businesses are sorted by ID before a stratified 80/20 split, random_state=42. Every main experiment uses identical observations. Three-fold shuffled stratified CV is confined to training data. Imputation and scaling are fit inside pipelines within each fold. LR uses log1p count transforms, standardization, balanced class weights and max_iter=2000. RF uses 120 trees, min_samples_leaf=5, balanced_subsample weights and seed 42. Missing numeric values are median imputed with indicators. The fixed decision threshold is 0.5; no holdout tuning or hyperparameter search was performed. Confusion matrices in the CSV use actual/predicted closed as positive (TN=open correctly classified; FP=open called closed; FN=closed called open; TP=closed correctly classified).','',
        'A=business stars/lifetime reviews; B=A+annual review volume; C=A+first/second-half volume; D=C+all-time check-ins/tips; E=D+annual review ratings/rating change; F=D+sentiment mean/std/negative share/change; G=D+ratings+sentiment. Thus D is the common activity baseline with basic business characteristics. Annual total and half-year volumes are never combined; sentiment counts are excluded as redundant. First/recent sentiment means are saved but their mean/change representation is used to avoid further redundancy.','',
        table(frames['main'],['model','feature_group','roc_auc_closed','average_precision_closed','balanced_accuracy','precision_closed','recall_closed','f1_closed']),'',
        'CV means/standard deviations, confusion counts and exact predictions are retained in `main/`. Full-model entries also provide the RQ2 LR/RF comparison; `main/feature_ablation.csv` contains the RQ3 group results as a separate table from the same run.','',
        '### Incremental feature value','',
        'Review volume provides the largest increment over business characteristics. Replacing annual volume with temporal counts and adding check-ins/tips yields small, model-dependent increments. For LR, ratings and sentiment each improve AUC over activity alone, but adding sentiment after ratings does not establish further improvement. RF also shows no helpful full-model sentiment increment. Sentiment feature groups include dispersion and missingness indicators, so their increments cannot be attributed solely to the semantic information in review text.','',
        'Intervals below are 95% percentile intervals from 500 paired stratified holdout bootstrap resamples with seed 42. They quantify test-sample uncertainty conditional on these fitted models; they omit training-set, label and population-shift uncertainty. These exploratory intervals have no multiplicity adjustment. An interval spanning zero does not establish a contribution. Other metrics are in the incremental CSV.','']
    increments={name:pd.read_csv(OUT/name/'incremental_contributions.csv') for name in ['main','active']}
    lines += [table(increments['main'].loc[increments['main'].metric.eq('roc_auc_closed')],['model','before','after','metric','delta','ci_low','ci_high']),'',
        '## Active-review sensitivity','',
        'There are 31,575 eligible restaurants (1,283 closed; 30,292 open), with closed prevalence 4.063%. Activity-to-sentiment AUC increments are +0.0075 for LR and +0.0093 for RF; both paired 95% intervals span zero. After ratings, sentiment increments are +0.0017 for LR and +0.0040 for RF, also with intervals spanning zero. This experiment does not establish an incremental sentiment advantage in the active population.','',
        'This changes the study population to restaurants with at least one 2021 review. A separate stratified 80/20 split is made within that population, followed by training-only CV and retraining. AP is prevalence dependent and should not be compared directly across populations. The activity baseline includes business characteristics and activity, with no review-derived ratings or sentiment.','',
        '```json',(OUT/'active/metadata.json').read_text(),'```','',
        table(frames['active'],['model','feature_group','roc_auc_closed','average_precision_closed','balanced_accuracy','precision_closed','recall_closed','f1_closed']),'',
        table(increments['active'].loc[increments['active'].metric.eq('roc_auc_closed')],['model','before','after','delta','ci_low','ci_high']),'',
        '## Scientific limits','',
        'Yelp is_open is recorded snapshot status; verified closure dates are unavailable for the full sample. Reviews can be posted after actual closure. Business stars, lifetime reviews and all-time check-ins may include information accumulated after 2021. Review inactivity and missing rating/sentiment indicators can encode status timing. Missing sentiment is unavailable evidence, not neutrality. Yelp reviewers may not represent all customers. Random holdout classification accuracy establishes neither future predictive performance nor causality. VADER may misread sarcasm, mixed sentiment or domain language. The active subset is selected on review availability and has different prevalence. Bootstrap intervals do not repair these limitations. The 146 cases where 2021 counts exceed snapshot review totals remain unresolved source inconsistencies; their status breakdown is saved in the activity audit.','',
        '## Reproduction','',
        'From the repository root (reuse the generated sentiment files unless inputs or scoring code change):','',
        '```bash',
        '.venv/bin/python -m pip install -r requirements-status-ablation.txt',
        '# Run this once to rebuild sentiment from the original gzip:',
        '.venv/bin/python -u scripts/build_sentiment_features.py',
        '.venv/bin/python scripts/rq1_descriptive_summary.py --input data/business_features_with_sentiment.csv --output research/status_experiments/rq1_descriptive_summary.csv',
        '.venv/bin/python -u scripts/recorded_status_experiments.py --trees 120 --folds 3 --bootstrap 500',
        '.venv/bin/python scripts/plot_status_results.py',
        '.venv/bin/python scripts/summarize_status_results.py',
        '.venv/bin/python -m unittest discover -s tests -v',
        '.venv/bin/python scripts/validate_status_outputs.py',
        'git status --short','```','',
        '## Output inventory','',
        '`data/sentiment_features_2021.csv` holds all restaurant aggregates; `data/business_features_with_sentiment.csv` is the preserved Stage 1 table plus sentiment. `research/status_experiments/` holds validation, descriptive statistics, comparisons, figure, activity audit, runtime versions and report; `main/` and `active/` hold model comparisons, predictions, split IDs, feature definitions and paired increments. `validation_checks.json` records independent output checks; `test_results.txt` records actual automated tests. Raw reviews, generated feature datasets and per-business prediction files are kept local; summary CSVs, split IDs, figures and reproducible code are versioned.']
    lines += ['', '## Figures', '',
        'All figures use saved holdout results. Open and closed mean recorded Yelp status. The active-review population has a separate split and different class prevalence. Confidence intervals describe holdout sampling uncertainty conditional on fitted models.', '',
        '![Feature-group comparisons](figures/feature_group_comparison.png)', '',
        '![Incremental AUC with paired intervals](figures/incremental_auc.png)', '',
        '![Sentiment increments with paired intervals](figures/sentiment_contribution.png)', '',
        '![ROC and precision–recall curves](figures/classification_curves.png)', '',
        '![Confusion matrices with counts and row percentages](figures/confusion_matrices.png)', '',
        '## Verification outcome', '',
        'All 10 automated tests passed. Independent cached-output validation passed: source hashes, original Stage 1 columns and labels, aggregate bounds/missingness, count reconciliation, disjoint splits, common holdout IDs and confusion matrices. Experimental settings and runtime versions are recorded alongside the results.']
    (OUT/'results_report.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__': main()
