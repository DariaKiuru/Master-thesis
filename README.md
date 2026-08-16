# Master Thesis Empirical Project

## Thesis title

**Investor Sentiment and Market Volatility during the Ukraine Crisis: FinBERT-Based Sentiment and GARCH Analysis of Selected European Equity Markets**

This repository contains the Python code and empirical data pipeline for a Quantitative Finance Master's thesis.

## Research objective

The project examines whether Ukraine-war-related Reddit sentiment is associated with subsequent volatility in selected European equity markets during 2021–2023, and whether this relationship differs across markets.

## Research questions

**RQ1:** Is Reddit-based investor sentiment significantly associated with subsequent market volatility in selected European equity markets during the 2021–2023 Ukraine-crisis period?

**RQ2:** Does the relationship between Reddit-based investor sentiment and market volatility differ across the selected European equity markets?

**Hypothesis H1:** More negative Reddit-based investor sentiment is associated with higher subsequent market volatility.

## Empirical workflow

1. Collect Reddit posts from `r/investing`, `r/stocks`, and `r/StockMarket`.
2. Clean and filter posts for Ukraine-war-related financial discussion.
3. Score title + body text using `ProsusAI/finbert`.
4. Construct a daily sentiment index from post-level scores.
5. Retain daily Reddit post count as a separate attention measure.
6. Collect closing levels for five European equity indices.
7. Calculate daily log returns.
8. Estimate GARCH(1,1) models with Student-t innovations.
9. Align Reddit and market data by trading day.
10. Create one-day lagged sentiment, attention, volatility, and return variables.
11. Estimate separate OLS regressions with HAC/Newey-West standard errors for each market.
12. Generate thesis-ready tables and figures.

## Sample period

`2021-01-01` through `2023-12-31`

## Reddit sources

The final analysis uses three finance/market-oriented communities:

- `r/investing`
- `r/stocks`
- `r/StockMarket`

The three communities are pooled into the main Reddit sample. Each qualifying
post is equally weighted in the daily sentiment index; subreddits are not given
equal aggregate weights. The subreddit identifier is retained for every post
so sample composition can be reported descriptively.

Both post titles and post bodies (`selftext`) are used.

Only English-language posts are retained.

## Sentiment model

The project uses `ProsusAI/finbert` through the Hugging Face Transformers library.

For each post, FinBERT provides:
- positive probability
- neutral probability
- negative probability

Long posts are divided into smaller chunks. The positive, neutral, and negative probabilities are averaged across chunks before the post-level score is constructed.

The finalized Phase 4A implementation is `src/07_score_finbert.py`. It uses
deterministic sentence-aware conceptual chunks of approximately 30 words, with
a maximum of 120 conceptual chunks per post. Tokenizer-safeguard fragments do
not change the conceptual-chunk weights: fragment probabilities are first
reconstructed into one conceptual-chunk probability vector using word-count
weights, and only then are conceptual chunks averaged to the post level. Silent
tokenizer truncation is not permitted.

The validated Phase 4A reconciliation is:

```text
34,879 retained conceptual chunks
-> 34,884 FinBERT model inputs
-> 34,879 reconstructed conceptual probability vectors
-> 1,503 post-level observations
```

The production scorer uses deterministic length-aware batches of 8 inputs,
dynamic padding, inference mode, and atomic checkpoints every 800 model inputs.
It skips inference when complete outputs validate successfully. To resume an
interrupted run from a matching checkpoint, use:

```powershell
python src/07_score_finbert.py --resume --batch-size 8 --checkpoint-interval 800
```

The canonical post-level output is
`data/processed/reddit_posts_finbert.csv`. Phase 4A diagnostics are stored as
`outputs/diagnostics/finbert_sentiment_summary.csv`,
`outputs/diagnostics/finbert_class_distribution.csv`, and
`outputs/diagnostics/finbert_review_sample.csv`.

`finbert_development_sample.csv` and `finbert_inference_benchmark.csv` are
development-only reproducibility artifacts. They are not empirical inputs and
do not contribute observations to the canonical Phase 4A output.

The post-level sentiment score is:

`sentiment_score = positive_prob - negative_prob`

The score ranges from `-1` to `+1`.

The daily sentiment index is the equal-weight mean of all post-level scores observed on that day.

Daily Reddit post count is retained separately as an attention measure.

The finalized Phase 4B implementation is
`src/08_build_daily_sentiment.py`. It validates the immutable Phase 4A input
before constructing `data/processed/daily_reddit_sentiment.csv` on the complete
1,095-day calendar from 2021-01-01 through 2023-12-31. Calendar days without a
relevant post retain `post_count = 0` and missing sentiment and mean
probabilities; sentiment is not imputed. Daily positive, neutral, and negative
label counts and mean FinBERT probabilities are descriptive supporting
variables, not alternative sentiment indices. The reproducibility summary is
`outputs/diagnostics/daily_reddit_sentiment_summary.csv`.

The calendar-day file is used for descriptive analysis. Later market alignment
must use `data/processed/reddit_posts_finbert.csv`, map post-level observations
to each index's next available trading day, and aggregate the mapped posts. It
must not average already-aggregated calendar-day sentiment means.

## Equity markets

| Market | Data source | Source symbol | Stored ticker | Price field |
|---|---|---|---|---|
| EURO STOXX 50 | Yahoo Finance | `^STOXX50E` | `^STOXX50E` | Adj Close |
| DAX | Yahoo Finance | `^GDAXI` | `^GDAXI` | Adj Close |
| CAC 40 | Yahoo Finance | `^FCHI` | `^FCHI` | Adj Close |
| FTSE 100 | Yahoo Finance | `^FTSE` | `^FTSE` | Adj Close |
| WIG20 | Stooq | `wig20` | `WIG20` | Close |

The source-specific price fields are standardized as `close_level` in the
common market-price dataset.

### WIG20 data provenance

The WIG20 observations were produced by Stooq. Because the live Stooq endpoint
returned a JavaScript-verification page, the dataset was retrieved through an
archived January 2025 copy of Stooq's historical-data response:

`https://web.archive.org/web/20250114102640id_/https://stooq.com/q/d/l/?s=wig20&i=d`

The diagnostic summary records `data_source = Stooq` and
`retrieval_method = Internet Archive snapshot of Stooq` for WIG20.

Daily log returns are calculated as:

`log_return_t = ln(P_t / P_t-1)`

The approved Phase 5 implementation is
`src/10_build_market_volatility.py`. Its canonical output is
`data/processed/market_returns_garch.csv`. It retains `log_return` in decimal
units and uses `return_pct = 100 * log_return` as the GARCH estimation input.

## Volatility model

Conditional volatility is estimated separately for each market using:
- constant mean
- GARCH(1,1)
- Student-t innovations

The Python `arch` package is used for estimation.

The estimated conditional standard deviation is retained as
`garch_volatility` in percentage-return units. Model parameters and truthful
optimizer/convergence diagnostics are saved under `outputs/tables/` and
`outputs/diagnostics/`, respectively. Information criteria are retained only
as technical diagnostics and are not used for model selection.

Phase 5 is frozen at commit
`f720e32fc73a79b7cc36d14223661d8055a04681`.

## Trading-day alignment

The Phase 6 implementation is `src/11_build_trading_day_alignment.py`, and its
canonical full-calendar output is
`data/processed/market_aligned_lagged.csv`. For each market separately, it
starts from post-level observations in `reddit_posts_finbert.csv`, maps every
post to the same or next actual market trading date, and then calculates the
equal-post-weight sentiment mean and post-count attention. A trading date with
no mapped posts retains `attention = 0` and missing sentiment; sentiment is not
imputed. Posts without a later in-sample trading date are retained in the
mapping reconciliation as `terminal_unmapped`.

Sentiment, attention, volatility, and return lags are created only after the
market-specific aggregation is merged onto the complete market panel.
`return_lag1` uses decimal `log_return`. The resulting regression-eligibility
rates are approximately 52.7% to 53.5% of finite-volatility rows because prior
trading-day sentiment is structurally missing when no qualifying post was
mapped to that day. Phase 6 has been reviewed and approved without changing
the frozen methodology. Phase 6 is frozen at commit
`d69ea18acdedd042efe3edc113fce3e593de7031`.

## Regression specification

The final regression is estimated separately for each market.

Dependent variable:

`garch_volatility_t`

Regressors:
- `sentiment_lag1`
- `attention_lag1`
- `volatility_lag1`
- `return_lag1`
- intercept

Inference uses HAC/Newey-West standard errors with a default maximum lag of 5.

The coefficient on `sentiment_lag1` is the main coefficient of interest.

Because higher sentiment values indicate more positive sentiment, H1 predicts a negative sentiment coefficient.

The Phase 7 implementation is `src/12_run_hac_regressions.py`. It estimates
exactly one intercept-including OLS model per market with
`garch_volatility` as the dependent variable and `sentiment_lag1`,
`attention_lag1`, `volatility_lag1`, and decimal `return_lag1` as regressors.
Reported inference uses HAC/Newey-West standard errors with maximum lag 5.
The approved sample sizes are 401 observations for EURO STOXX 50, 406 for DAX,
407 for CAC 40, 397 for FTSE 100, and 402 for WIG20. The design is
associational; lagging predictors does not establish causality.

The lagged-sentiment point estimate is negative for EURO STOXX 50, DAX, CAC 40,
and FTSE 100 and positive for WIG20. All five 95% confidence intervals include
zero, so none of the sentiment coefficients is statistically significant under
the frozen specification. Because eligibility requires observable prior-day
sentiment, every eligible observation has positive `attention_lag1`; the
attention coefficient therefore describes variation in discussion intensity
conditional on discussion being observed, not zero versus positive discussion.
The regression samples are concentrated in 2022. Phase 7 has been validated,
reviewed, and approved by the researcher without a specification search or
formal cross-market coefficient-equality test.

Phase 7 is frozen at commit
`b032fac7ea05cfa575669395cec983b72799fd47`. The five sentiment coefficients
are not statistically significant under the approved specification; this does
not authorize alternative specifications or a robustness search.

## Final thesis-output consolidation

Phase 8 is implemented in `src/13_build_final_thesis_outputs.py`. It verifies
the frozen canonical hashes and consolidates existing evidence; it does not
estimate or rerun an empirical model. Its principal outputs are:

- `outputs/tables/final_thesis_output_manifest.csv`
- `outputs/tables/final_sample_overview.csv`
- `outputs/tables/final_regression_table.csv`
- `outputs/tables/final_sentiment_results_summary.csv`
- `outputs/tables/research_question_summary.csv`
- `outputs/tables/final_methodological_limitations.csv`
- `outputs/tables/active_script_inventory.csv`
- `outputs/diagnostics/final_reproducibility_audit.csv`
- `outputs/final_results_review.md`

The frozen `outputs/tables/regression_results.csv` remains the authoritative
machine-readable Phase 7 regression table. The Phase 8 regression table is a
readability-oriented derivative containing coefficients, HAC standard errors,
sample sizes, R-squared values, and the fixed maximum HAC lag.

## Methods intentionally excluded

The final active methodology does **not** include:
- wavelet analysis
- event-study analysis
- VAR
- Granger causality
- robustness-analysis specifications
- alternative sentiment indices
- FinBERT fine-tuning
- machine-learning market-prediction models

## Repository structure

The validated upstream empirical pipeline is
`src/01_download_market_data.py` through `src/08_build_daily_sentiment.py`,
including the intermediate Reddit inspection and relevance-validation scripts.
The one-time descriptive-results backfill is implemented in
`src/09_build_descriptive_results.py`, Phase 5 return/GARCH construction is
implemented in `src/10_build_market_volatility.py`, and the approved Phase 6
alignment is implemented in `src/11_build_trading_day_alignment.py`. The
approved Phase 7 regressions are implemented in
`src/12_run_hac_regressions.py`. Final reporting consolidation and the
reproducibility audit are implemented in
`src/13_build_final_thesis_outputs.py`.

```text
Master-thesis/
├── AGENTS.md
├── README.md
├── requirements.txt
├── .gitignore
├── config.py
├── data/
│   ├── raw/
│   │   ├── market/
│   │   └── reddit/
│   └── processed/
├── src/
│   ├── 01_download_market_data.py
│   ├── 02_extract_reddit.py
│   ├── 03_inspect_reddit_candidates.py
│   ├── 04_validate_reddit_relevance.py
│   ├── 05_compare_reddit_relevance_rules.py
│   ├── 06_clean_reddit.py
│   ├── 07_score_finbert.py
│   ├── 08_build_daily_sentiment.py
│   ├── 09_build_descriptive_results.py
│   ├── 10_build_market_volatility.py
│   ├── 11_build_trading_day_alignment.py
│   ├── 12_run_hac_regressions.py
│   └── 13_build_final_thesis_outputs.py
├── outputs/
│   ├── figures/
│   ├── tables/
│   └── diagnostics/
└── archive/
    ├── legacy_scripts/
    └── test_outputs/
```

Legacy scripts and experimental outputs remain archived and are not part of
the active numbered workflow.

## Thesis outputs

### 5.1 Sample and Descriptive Statistics
- sample counts
- descriptive statistics of Reddit sentiment and attention
- descriptive statistics of market returns and GARCH volatility

### 5.2 Reddit Sentiment and Attention
- daily Reddit sentiment plot
- daily Reddit post-count plot
- distribution of FinBERT positive, neutral, and negative classifications

### 5.3 Conditional Volatility in European Markets
- GARCH(1,1)-Student-t parameter table
- conditional-volatility plot across the five markets

### 5.4 Sentiment-Volatility Regression Results
- regression table with one column per market
- HAC standard errors
- p-values and confidence intervals
- cross-market comparison of the lagged sentiment coefficient

## Active script order

Run the numbered scripts in order only when a full authorized reproduction is
required:

1. `01_download_market_data.py` - market-price collection (network; frozen)
2. `02_extract_reddit.py` - Reddit candidate extraction (network; frozen)
3. `03_inspect_reddit_candidates.py` - candidate-corpus audit
4. `04_validate_reddit_relevance.py` - initial relevance-rule dry run
5. `05_compare_reddit_relevance_rules.py` - relevance-rule comparison
6. `06_clean_reddit.py` - final cleaned and filtered Reddit corpus
7. `07_score_finbert.py` - post-level FinBERT inference (expensive; frozen)
8. `08_build_daily_sentiment.py` - complete calendar-day Reddit series
9. `09_build_descriptive_results.py` - validated descriptive tables and figures
10. `10_build_market_volatility.py` - returns and GARCH volatility (frozen)
11. `11_build_trading_day_alignment.py` - market-specific mapping and lags (frozen)
12. `12_run_hac_regressions.py` - five approved OLS-HAC regressions (frozen)
13. `13_build_final_thesis_outputs.py` - deterministic Phase 8 consolidation and audit

The full input/output and rerun guidance is recorded in
`outputs/tables/active_script_inventory.csv`. In normal final-stage use, verify
the frozen hashes and rerun only script 13. Do not redownload market or Reddit
data, rerun FinBERT, or re-estimate frozen models merely to reproduce the final
reporting package.

An empirical phase is complete only after its canonical output, technical
validation diagnostics, descriptive or thesis-facing tables, appropriate
descriptive figures, and a review of observed patterns and unusual features
have been produced. These reporting requirements must describe approved
variables without adding new empirical methodology.

## Reproducibility

The final repository should:
- use paths relative to the repository root;
- centralize methodological settings in `config.py`;
- keep generated outputs outside the repository root;
- avoid committing virtual environments and caches;
- retain validation checks for data ranges, duplicates, probabilities, GARCH outputs, and regression inputs;
- allow expensive steps such as Reddit extraction and FinBERT inference to be skipped when valid outputs already exist;
- verify frozen Phase 4A, Phase 5, Phase 6, and Phase 7 hashes before Phase 8 consolidation;
- record final repository and output checks in
  `outputs/diagnostics/final_reproducibility_audit.csv`.

## Status

The retrospective descriptive-results checkpoint for the validated market-data,
Reddit-cleaning, FinBERT, and daily Reddit phases is complete. Phase 5 market
returns and the five constant-mean GARCH(1,1)-Student-t models have been computed,
validated, and approved by the researcher, with canonical data, tables,
diagnostics, and figures under the repository's standard output directories.
Phase 6 market-specific trading-day alignment and lag construction have been
computed, validated, and approved by the researcher. The regression
sample is conditional on observable prior-trading-day Reddit sentiment and is
strongly concentrated in 2022. Phase 7's five separate OLS-HAC regressions have
been computed, validated, reviewed, and approved by the researcher. No
lagged-sentiment coefficient is statistically significant under the frozen
specification. Phase 8 final thesis-output consolidation, interpretation
synthesis, and reproducibility audit have been performed without adding or
rerunning an empirical model. The uncommitted Phase 8 working-tree changes
remain subject to researcher review.
