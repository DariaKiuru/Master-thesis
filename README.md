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

## Repository structure

The numbered tree below was an early plan. The active pipeline through Phase
4A is `src/01_download_market_data.py` through
`src/07_score_finbert.py`, including the intermediate Reddit inspection and
relevance-validation scripts. Later numbered stages are not yet finalized.

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

Phase 4A is complete: the immutable 1,503-post Phase 3B dataset has been scored
with FinBERT and validated at the post level. Daily aggregation and subsequent
market, GARCH, alignment, and regression phases are not part of this milestone.
