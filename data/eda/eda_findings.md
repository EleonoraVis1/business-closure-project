# Yelp Restaurant Operating Status: EDA Findings

## 1. Dataset overview

- Total restaurants: 52,268
- Open: 34,987 (66.94%)
- Closed: 17,281 (33.06%)

## 2. Largest open/closed differences

Features are ordered by the absolute value of Cohen's d. Positive values mean the open group has a higher mean.

- **reviews_recent_6m_2021**: open mean = 5.732, closed mean = 0.112; open median = 2.000, closed median = 0.000; Cohen's d = 0.629; Welch FDR q = 0.
- **reviews_2021**: open mean = 11.495, closed mean = 0.398; open median = 5.000, closed median = 0.000; Cohen's d = 0.628; Welch FDR q = 0.
- **reviews_recent_3m_2021**: open mean = 2.706, closed mean = 0.032; open median = 1.000, closed median = 0.000; Cohen's d = 0.614; Welch FDR q = 0.
- **reviews_first_6m_2021**: open mean = 5.763, closed mean = 0.286; open median = 2.000, closed median = 0.000; Cohen's d = 0.584; Welch FDR q = 0.
- **distinct_tip_users_2021**: open mean = 0.670, closed mean = 0.036; open median = 0.000, closed median = 0.000; Cohen's d = 0.497; Welch FDR q = 0.
- **tips_2021**: open mean = 0.698, closed mean = 0.037; open median = 0.000, closed median = 0.000; Cohen's d = 0.492; Welch FDR q = 0.
- **business_review_count**: open mean = 104.143, closed mean = 53.101; open median = 40.000, closed median = 24.000; Cohen's d = 0.272; Welch FDR q = 7.955e-306.
- **all_time_checkins**: open mean = 193.132, closed mean = 101.732; open median = 52.000, closed median = 33.000; Cohen's d = 0.198; Welch FDR q = 3.035e-165.
- **avg_review_stars_2021**: open mean = 3.436, closed mean = 3.587; open median = 3.750, closed median = 4.000; Cohen's d = -0.124; Welch FDR q = 0.0001398.
- **avg_review_stars_recent_6m**: open mean = 3.440, closed mean = 3.571; open median = 3.778, closed median = 4.000; Cohen's d = -0.101; Welch FDR q = 0.02812.

## 3. Strongest feature relationships

- tips_2021 and distinct_tip_users_2021: Spearman rho = 0.999.
- reviews_2021 and reviews_recent_6m_2021: Spearman rho = 0.939.
- reviews_2021 and reviews_first_6m_2021: Spearman rho = 0.924.
- reviews_recent_6m_2021 and reviews_recent_3m_2021: Spearman rho = 0.902.
- avg_review_stars_2021 and avg_review_stars_recent_6m: Spearman rho = 0.868.
- avg_review_stars_2021 and avg_review_stars_first_6m: Spearman rho = 0.868.
- reviews_2021 and reviews_recent_3m_2021: Spearman rho = 0.845.
- business_review_count and all_time_checkins: Spearman rho = 0.839.
- reviews_first_6m_2021 and reviews_recent_6m_2021: Spearman rho = 0.785.
- reviews_first_6m_2021 and reviews_recent_3m_2021: Spearman rho = 0.710.

## 4. Interpretation and limitations

- The target is Yelp's recorded operating status, not a verified closure date or future closure event.
- These are observational comparisons and do not establish causation.
- Count features are highly likely to be skewed; log1p plots are included to make their distributions easier to inspect.
- Missing review averages are not zero-star ratings. They should remain missing during descriptive analysis.
- All-time check-ins have a different time window from 2021 reviews and tips.
- Statistical significance should not be confused with practical importance. Examine effect sizes and distributions.
- Any imputation, scaling, encoding, or feature selection for classification must be fitted using training data only.

## 5. Questions for the next stage

1. Which features show the largest and most consistent differences between open and closed restaurants?
2. Are review counts, check-ins, and tips strongly redundant?
3. Do ratings provide information beyond customer activity?
4. How should missing rating averages be handled in a model pipeline?
5. Which feature groups should be compared in the classification experiments?