# AGENTS.md

## Project purpose

This repository contains the empirical Python work for a Quantitative Finance Master's thesis:

**Investor Sentiment and Market Volatility during the Ukraine Crisis: FinBERT-Based Sentiment and GARCH Analysis of Selected European Equity Markets**

The empirical sample covers **2021-01-01 through 2023-12-31**.

The project should remain simple, reproducible, transparent, and understandable to a researcher with basic Python knowledge.

## Final empirical workflow

Reddit posts  
→ cleaning and Ukraine-war relevance filtering  
→ FinBERT sentiment inference  
→ post-level sentiment scores  
→ daily sentiment index and Reddit attention  
→ European equity-index prices  
→ daily log returns  
→ GARCH(1,1) with Student-t innovations  
→ trading-day alignment and lagged variables  
→ separate sentiment-volatility regressions for five indices  
→ thesis-ready tables and figures

Do not introduce additional empirical methods without explicit user instruction.

## Methods that are NOT part of the final thesis

Do not add or restore:

- wavelet analysis
- event-study analysis
- VAR
- Granger causality
- robustness-analysis specifications
- alternative sentiment indices
- Fama-French models
- GARCH-X
- machine-learning market-prediction models
- SVM
- FinBERT fine-tuning
- BERT training

Old experimental code using these ideas may be archived, but it must not become part of the active pipeline.

## Research questions

**RQ1:** Is Reddit-based investor sentiment significantly associated with subsequent market volatility in selected European equity markets during the 2021–2023 Ukraine-crisis period?

**RQ2:** Does the relationship between Reddit-based investor sentiment and market volatility differ across the selected European equity markets?

**H1:** More negative Reddit-based investor sentiment is associated with higher subsequent market volatility.

Because the sentiment index increases with more positive sentiment, H1 implies an expected **negative coefficient on lagged sentiment**.

## Reddit data

Use these three finance-focused Reddit communities:

- `r/investing`
- `r/stocks`
- `r/StockMarket`

The three communities are pooled into one Reddit sample.

Retain the subreddit identifier for every post so that sample composition can
be reported descriptively.

Each qualifying post receives equal weight in the daily sentiment index,
irrespective of subreddit.

Do not balance, resample, or assign fixed weights to subreddits.

Reddit sentiment should be interpreted as sentiment expressed in the selected
finance-oriented Reddit communities rather than as representative sentiment
of all investors.

Final period:

- `2021-01-01` through `2023-12-31`

Use both:

- `title`
- `selftext`

Use English-language posts only.

The final relevance filter must focus on Ukraine-war-related financial discussion. Generic mentions of oil, gas, energy, inflation, or recession alone must not automatically make a post relevant.

## FinBERT

Use `ProsusAI/finbert` through Hugging Face Transformers.

Do not retrain or fine-tune the model.

Long title + selftext observations are chunked using the methodology currently documented in the thesis:

- approximately 30 words per chunk
- maximum 120 chunks per post

For every chunk retain all three FinBERT probabilities:

- positive
- neutral
- negative

For a multi-chunk post, average each probability across chunks first.

Post-level sentiment is:

`sentiment_score = positive_prob - negative_prob`

The valid range is:

`-1 <= sentiment_score <= 1`

For descriptive labels only, assign the class with the largest averaged probability.

Do not use arbitrary `+/- 0.05` classification thresholds.

## Daily sentiment and attention

Daily sentiment is the equal-weight arithmetic mean of all post-level sentiment scores assigned to that day:

`S_t = mean(sentiment_score)`

Daily attention is the number of relevant Reddit posts:

`A_t = post_count`

Sentiment and attention are separate constructs.

Do not assign `sentiment = 0` to days with no Reddit posts. No discussion is not the same as neutral discussion.

## Equity markets

Use these five indices and sources:

- EURO STOXX 50: Yahoo Finance, `^STOXX50E`
- DAX: Yahoo Finance, `^GDAXI`
- CAC 40: Yahoo Finance, `^FCHI`
- FTSE 100: Yahoo Finance, `^FTSE`
- WIG20: Stooq, symbol `wig20`, standardized as `WIG20`

The WIG20 observations were produced by Stooq and retrieved through an archived
January 2025 response from Stooq's historical-data endpoint because the live
endpoint returned a JavaScript-verification page. Preserve this provenance in
the market-data diagnostics.

Do not use `WIG20.WA` or `GPW.WA` as WIG20.

For the four Yahoo Finance indices, use adjusted closing levels. For WIG20, use
the Stooq daily Close field. Store both consistently in the common
`close_level` variable.

Calculate daily log returns as:

`log_return_t = ln(P_t / P_t-1)`

## GARCH

Estimate one model separately for each index.

Use:

- constant mean
- GARCH(1,1)
- Student-t innovations

Use the Python `arch` package unless explicitly instructed otherwise.

The final market-volatility variable is the estimated GARCH conditional standard deviation.

Do not compare many alternative GARCH specifications unless explicitly asked.

## Regression

Estimate the final model separately for each index.

Dependent variable:

`garch_volatility_t`

Regressors:

- `sentiment_lag1`
- `attention_lag1`
- `volatility_lag1`
- `return_lag1`
- intercept

Use OLS with HAC/Newey-West standard errors.

Default HAC maximum lag: `5`

The main coefficient of interest is the coefficient on `sentiment_lag1`.

Do not describe statistical significance as proof of structural causality.

## Trading-day alignment

Keep the original calendar-day Reddit series for descriptive analysis.

For regressions:

- map non-trading-day Reddit observations to the next available trading day;
- aggregate mapped data from post-level observations;
- merge with each index's actual trading dates;
- create lags only after the trading-day alignment;
- never use future information.

## Thesis outputs

The active pipeline must produce material for:

### 5.1 Sample and Descriptive Statistics
- Reddit sample counts
- descriptive statistics of sentiment and attention
- descriptive statistics of returns and GARCH volatility

### 5.2 Reddit Sentiment and Attention
- daily sentiment plot
- daily Reddit post-count plot
- positive / neutral / negative FinBERT classification distribution

### 5.3 Conditional Volatility in European Markets
- GARCH(1,1)-Student-t parameter table
- conditional-volatility figure across the five indices

### 5.4 Sentiment-Volatility Regression Results
- one regression table with five market columns
- coefficient, HAC standard error, p-value and confidence interval
- cross-market comparison of the sentiment coefficient

Do not create event-study, robustness, Granger, VAR, or wavelet result sections.

## Coding style

Prioritize readability over software-engineering complexity.

Prefer:
- small functions
- `pathlib.Path`
- pandas DataFrames
- descriptive names
- clear comments
- simple scripts
- explicit validation checks

Avoid unless genuinely necessary:
- custom classes
- databases
- Docker
- workflow orchestration frameworks
- MLflow
- complex configuration systems
- unnecessary abstractions

Important methodological settings should be centralized in `config.py`.

## Data validation

Check at minimum:
- FinBERT probabilities are within `[0, 1]`
- `positive_prob + neutral_prob + negative_prob` is approximately `1`
- `sentiment_score` is within `[-1, 1]`
- market `close_level` values are positive
- there are no duplicate `index_name + date` combinations
- GARCH conditional volatility is positive and finite
- regression input contains no NaN or infinite values
- all final observations fall inside the thesis sample period

## Repository hygiene

Do not commit:
- `.venv/`
- `venv/`
- `__pycache__/`
- `*.pyc`
- `.DS_Store`
- `python-install.log`
- model caches
- temporary checkpoints

Do not place generated analysis files in the repository root.

Keep final outputs under:
- `outputs/tables/`
- `outputs/figures/`
- `outputs/diagnostics/`

Preserve useful experimental work under `archive/` until the replacement pipeline has been verified.

## Working rules for Codex

Before editing:
1. inspect the relevant existing files;
2. reuse working logic where appropriate;
3. identify conflicts between existing code and this methodology;
4. do not silently change methodological assumptions.

For every implementation task:
- change only the requested phase unless another change is strictly required;
- run appropriate validation or smoke tests;
- report files created, modified, moved, or removed;
- report commands executed;
- report unresolved issues;
- do not fabricate successful results when a network request or dependency fails.

When methodology and existing code conflict, follow this `AGENTS.md` and the current user task rather than legacy experimental code.
