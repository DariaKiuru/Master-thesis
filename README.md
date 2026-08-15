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

1. Collect Reddit posts from `r/investing` and `r/stocks`.
2. Clean and filter posts for Ukraine-war-related financial discussion.
3. Score title + body text using `ProsusAI/finbert`.
4. Construct a daily sentiment index from post-level scores.
5. Retain daily Reddit post count as a separate attention measure.
6. Collect adjusted close prices for five European equity indices.
7. Calculate daily log returns.
8. Estimate GARCH(1,1) models with Student-t innovations.
9. Align Reddit and market data by trading day.
10. Create one-day lagged sentiment, attention, volatility, and return variables.
11. Estimate separate OLS regressions with HAC/Newey-West standard errors for each market.
12. Generate thesis-ready tables and figures.

## Sample period

`2021-01-01` through `2023-12-31`

## Reddit sources

The final analysis uses:
- `r/investing`
- `r/stocks`

Both post titles and post bodies (`selftext`) are used.

Only English-language posts are retained.

## Sentiment model

The project uses `ProsusAI/finbert` through the Hugging Face Transformers library.

For each post, FinBERT provides:
- positive probability
- neutral probability
- negative probability

Long posts are divided into smaller chunks. The positive, neutral, and negative probabilities are averaged across chunks before the post-level score is constructed.

The post-level sentiment score is:

`sentiment_score = positive_prob - negative_prob`

The score ranges from `-1` to `+1`.

The daily sentiment index is the equal-weight mean of all post-level scores observed on that day.

Daily Reddit post count is retained separately as an attention measure.

## Equity markets

| Market | Yahoo Finance ticker |
|---|---|
| EURO STOXX 50 | `^STOXX50E` |
| DAX | `^GDAXI` |
| CAC 40 | `^FCHI` |
| FTSE 100 | `^FTSE` |
| WIG20 | `WIG20.WA` |

Adjusted closing prices are used.

Daily log returns are calculated as:

`log_return_t = ln(P_t / P_t-1)`

## Volatility model

Conditional volatility is estimated separately for each market using:
- constant mean
- GARCH(1,1)
- Student-t innovations

The Python `arch` package is used for estimation.

The estimated conditional standard deviation is retained as the daily market-volatility variable.

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

## Planned repository structure

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
│   ├── 03_clean_reddit.py
│   ├── 04_score_finbert.py
│   ├── 05_build_daily_sentiment.py
│   ├── 06_fit_garch.py
│   ├── 07_build_analysis_dataset.py
│   ├── 08_descriptive_results.py
│   ├── 09_regression_analysis.py
│   └── run_pipeline.py
├── outputs/
│   ├── figures/
│   ├── tables/
│   └── diagnostics/
└── archive/
    ├── legacy_scripts/
    └── test_outputs/
```

The repository is being refactored toward this structure. Legacy scripts and experimental outputs should be archived rather than immediately deleted.

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

## Current development workflow

The repository is being cleaned and implemented in phases:

1. repository audit
2. repository structure and cleanup
3. market-data pipeline
4. Reddit extraction and cleaning
5. FinBERT scoring
6. daily sentiment construction
7. GARCH modelling
8. trading-day alignment and regression
9. thesis-ready outputs and final reproducibility checks

Each phase should be reviewed before moving to the next one.

## Reproducibility

The final repository should:
- use paths relative to the repository root;
- centralize methodological settings in `config.py`;
- keep generated outputs outside the repository root;
- avoid committing virtual environments and caches;
- retain validation checks for data ranges, duplicates, probabilities, GARCH outputs, and regression inputs;
- allow expensive steps such as Reddit extraction and FinBERT inference to be skipped when valid outputs already exist.

## Status

This repository is currently being refactored from an experimental development structure into the final thesis pipeline. The code and file structure may change as individual implementation phases are completed.
