# Final results review

This is a researcher-facing synthesis of the frozen empirical evidence. It is
not a finished thesis chapter and introduces no additional empirical model.

## 1. Sample construction

The validated Phase 3A corpus contains 3,033 candidate posts. The final pooled
sample contains 1,503 equally weighted posts from r/investing, r/stocks, and
r/StockMarket. Of these, 1,197 (79.6%) occur in 2022, compared
with 83 in 2021 and 223 in 2023. The subreddit counts are 772 for r/stocks, 433
for r/StockMarket, and 298 for r/investing.

## 2. Sentiment descriptives

FinBERT's descriptive argmax labels comprise 106
positive, 953 neutral, and
444 negative posts. Neutral labels therefore
dominate. The equal-post-weight mean sentiment score is
-0.132582. The complete 1,095-day calendar contains
504 populated sentiment days and
591 zero-post days. Sentiment remains missing on
zero-post days. The mean across populated calendar days is
-0.107822; it differs from the post-level mean because it
weights populated days rather than individual posts.

## 3. Market returns and GARCH

WIG20 has the largest raw log-return standard deviation and the highest average
conditional volatility (1.446 percentage-return
standard-deviation units). FTSE 100 has the lowest corresponding raw return
variability and average conditional volatility (0.883).
The largest conditional-volatility readings occur in late February or March
2022. All five constant-mean GARCH(1,1)-Student-t models converged. Estimated
GARCH persistence (alpha + beta) is EURO STOXX 50 0.970, DAX 0.972, CAC 40 0.959, FTSE 100 0.877, WIG20 0.978; persistence is high,
especially for WIG20, DAX, EURO STOXX 50, and CAC 40.

## 4. Trading-day alignment and missingness

Regression alignment starts from the 1,503 post-level FinBERT observations, not
the calendar-day sentiment means. All posts map successfully to the same or next
actual trading date for every market (1,503 of
1,503 in each case). Market-calendar differences create some different mapping
dates. Across markets, 46.3% to
47.1% of trading days have zero
attention; aligned sentiment remains missing on those days. No sentiment
imputation is used. Phase 7 retains
52.7% to
53.5% of finite
volatility observations.

## 5. Regression results

The approved specification estimates five separate intercept-including OLS
models with HAC/Newey-West standard errors (maximum lag 5). `garch_volatility`
is the dependent variable; regressors are `sentiment_lag1`, `attention_lag1`,
`volatility_lag1`, and decimal `return_lag1`.

| Market | N | Sentiment coefficient | HAC SE | p-value | 95% CI |
|---|---:|---:|---:|---:|---:|
| EURO STOXX 50 | 401 | -0.008802327 | 0.030342176 | 0.771738191 | [-0.068271899, 0.050667246] |
| DAX | 406 | -0.011603800 | 0.024137895 | 0.630708725 | [-0.058913205, 0.035705606] |
| CAC 40 | 407 | -0.013180859 | 0.021938896 | 0.547974091 | [-0.056180305, 0.029818586] |
| FTSE 100 | 397 | -0.010544970 | 0.021448069 | 0.622965909 | [-0.052582414, 0.031492474] |
| WIG20 | 402 | 0.007913039 | 0.013885201 | 0.568752213 | [-0.019301454, 0.035127532] |

Four point estimates are negative, as predicted by H1, while WIG20's estimate
is positive. Every 95% confidence interval includes zero, and none of the five
sentiment coefficients is statistically significant. Lagged volatility is
strongly positive in every market (coefficients
0.827-0.913)
and statistically precise. Attention is positive in all five models
(0.0045-0.0080), but
its interpretation is conditional on positive prior-day discussion because
every eligible observation has `attention_lag1 > 0`. Approximately
55.2%
to 55.5%
of each estimation sample comes from 2022.

## 6. RQ1

The frozen market-specific OLS-HAC models do not provide statistically
significant evidence that lagged Reddit sentiment is associated with subsequent
GARCH conditional volatility in any of the five markets under the approved
specification.

## 7. RQ2

Point estimates differ across markets, including four negative estimates and
one positive WIG20 estimate, but all confidence intervals include zero and
overlap substantially. The results do not provide strong evidence of systematic
cross-market heterogeneity. No formal coefficient-equality test was part of the
frozen design.

## 8. H1

Four of five coefficients have the hypothesized negative sign, but none is
statistically significant. The results are directionally consistent with H1 in
four markets but do not provide statistically significant support for H1
overall.

## 9. Main limitations

- Sentiment represents three selected finance-oriented Reddit communities, not
  all investors.
- The Reddit corpus and regression samples are concentrated in 2022.
- There are 591 zero-post calendar days, and eligible regression samples are
  conditional on observed prior-aligned-trading-day sentiment.
- Attention varies only over positive values in the eligible samples.
- FinBERT is used pretrained without thesis-specific fine-tuning.
- Lagging predictors does not establish structural causation.
- Cross-market point estimates are not a formal coefficient-equality test.

## 10. Main reproducibility conclusions

The frozen canonical hashes match the approved checkpoints. Active scripts
compile, required inputs and configured output paths exist, market names and
units are consistent, archived experiments remain separate from the numbered
pipeline, and Git ignore rules cover environments, caches, local secrets, and
checkpoints. The final output inventory is in
`outputs/tables/final_thesis_output_manifest.csv`; detailed audit checks are in
`outputs/diagnostics/final_reproducibility_audit.csv`.

All statements above are associational. No causal claim, alternative
specification, robustness search, sentiment imputation, or new sentiment index
is used.
