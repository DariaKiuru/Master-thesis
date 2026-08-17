# Thesis Table and Figure Caption Catalogue

This catalogue provides a proposed thesis-facing numbering and placement scheme for the validated outputs in the frozen repository state beginning at commit `8d312c17bb193faaa3ef83b4882dfa7f9dfb0996`. It does not change, recalculate, or reinterpret any empirical result.

The recommended body is deliberately selective: six tables and five figures. Eleven supporting tables and five supporting figures receive appendix numbers. Forty-two remaining CSV artifacts are classified as unnumbered technical/QC material. The machine-readable catalogue is [thesis_table_figure_captions.csv](tables/thesis_table_figure_captions.csv).

Table numbers and titles should appear above tables. Figure captions should appear below figures. Notes and source lines should appear below both. Global numbering is the primary recommendation; chapter numbering is included only as an optional mapping.

Terminology throughout follows the frozen methodology: three finance-oriented Reddit communities, post-level FinBERT sentiment, daily Reddit sentiment, Reddit attention, GARCH conditional volatility measured as a conditional standard deviation, and associational OLS-HAC regression estimates.

# Recommended Main-Text Tables

## Table 1. Construction of the Final Reddit Sample

- Technical filename: `outputs/tables/reddit_sample_construction.csv`
- Recommended chapter: Chapter 3 — Data
- Optional chapter number: Table 3.1

Notes: The table traces the validated corpus from 3,033 candidate posts to 1,503 qualifying English-language posts from r/investing, r/stocks, and r/StockMarket. Each retained post enters the pooled sample with equal weight; neither years nor subreddits were artificially balanced.

Source: Author's calculations based on Reddit data.

## Table 2. Descriptive Statistics of Post-Level FinBERT Sentiment

- Technical filename: `outputs/tables/finbert_post_sentiment_descriptives.csv`
- Recommended chapter: Chapter 5 — Results
- Optional chapter number: Table 5.1

Notes: Sentiment is positive probability minus negative probability and ranges from -1 to 1. Statistics use 1,503 equally weighted posts from three finance-oriented Reddit communities. Positive, neutral, and negative labels are descriptive argmax classifications based on the averaged post-level probabilities.

Source: Author's calculations based on Reddit data and FinBERT sentiment scores.

## Table 3. Descriptive Statistics of Daily Reddit Sentiment and Attention

- Technical filename: `outputs/tables/daily_reddit_descriptives.csv`
- Recommended chapter: Chapter 5 — Results
- Optional chapter number: Table 5.2

Notes: Daily Reddit sentiment is the equal-weight mean of observed post-level FinBERT scores on each calendar day. Reddit attention is the daily count of qualifying posts. The calendar contains 1,095 days; zero-post days retain attention equal to zero and missing sentiment and are not treated as neutral.

Source: Author's calculations based on Reddit data and FinBERT sentiment scores.

## Table 4. Estimated GARCH(1,1)-Student-t Parameters

- Technical filename: `outputs/tables/garch_parameters.csv`
- Recommended chapter: Chapter 5 — Results
- Optional chapter number: Table 5.3

Notes: One constant-mean GARCH(1,1) model with Student-t innovations is estimated separately for each market using percentage log returns. The table reports the volatility-equation parameters and degrees of freedom; alpha plus beta summarizes estimated persistence.

Source: Author's calculations based on market data from Yahoo Finance and Stooq.

## Table 5. Descriptive Statistics of GARCH Conditional Volatility

- Technical filename: `outputs/tables/garch_volatility_descriptives.csv`
- Recommended chapter: Chapter 5 — Results
- Optional chapter number: Table 5.4

Notes: GARCH conditional volatility is the estimated conditional standard deviation from one constant-mean GARCH(1,1)-Student-t model per market. Values are reported in percentage-return standard-deviation units over each market's actual 2021–2023 trading dates.

Source: Author's calculations based on market data from Yahoo Finance and Stooq.

## Table 6. OLS-HAC Regression Results for Lagged Reddit Sentiment and Market Volatility

- Technical filename: `outputs/tables/regression_results.csv`
- Recommended chapter: Chapter 5 — Results
- Optional chapter number: Table 5.5

Notes: The dependent variable is GARCH conditional volatility measured as the conditional standard deviation in percentage-return units. Models are estimated separately for each market using OLS with HAC/Newey-West standard errors and maximum lag 5. Regressors are lagged Reddit sentiment, lagged Reddit attention, lagged conditional volatility, and the lagged decimal log return. Samples are conditional on observable prior-trading-day Reddit sentiment. Estimates are associational and should not be interpreted causally.

Source: Author's calculations based on Reddit data, FinBERT sentiment scores, and market data from Yahoo Finance and Stooq.

# Recommended Main-Text Figures

## Figure 1. Composition of the Final Reddit Sample

- Technical filename: `outputs/figures/reddit_sample_composition.png`
- Recommended chapter: Chapter 3 — Data
- Optional chapter number: Figure 3.1

Notes: The final sample contains 1,503 qualifying posts from r/investing, r/stocks, and r/StockMarket. Counts are shown by calendar year and subreddit. The sample is heavily concentrated in 2022; no balancing or fixed subreddit weighting was applied.

Source: Author's calculations based on Reddit data.

## Figure 2. Daily Reddit Sentiment during the 2021–2023 Sample Period

- Technical filename: `outputs/figures/daily_reddit_sentiment.png`
- Recommended chapter: Chapter 5 — Results
- Optional chapter number: Figure 5.1

Notes: Daily Reddit sentiment is the equal-weight mean of observed post-level FinBERT scores, where each score equals positive probability minus negative probability. Zero-post days retain missing sentiment and are not treated as neutral; no filling, interpolation, or smoothing is applied.

Source: Author's calculations based on Reddit data and FinBERT sentiment scores.

## Figure 3. Distribution of Post-Level FinBERT Classifications

- Technical filename: `outputs/figures/finbert_label_distribution.png`
- Recommended chapter: Chapter 5 — Results
- Optional chapter number: Figure 5.2

Notes: Bars show the descriptive argmax classification of the averaged post-level FinBERT probabilities for all 1,503 qualifying posts. The categories contain 106 positive, 953 neutral, and 444 negative posts. These labels are descriptive and are not alternative sentiment indices.

Source: Author's calculations based on Reddit data and FinBERT sentiment scores.

## Figure 4. Estimated GARCH Conditional Volatility across the Five European Equity Markets

- Technical filename: `outputs/figures/garch_conditional_volatility.png`
- Recommended chapter: Chapter 5 — Results
- Optional chapter number: Figure 5.3

Notes: Each panel shows the conditional standard deviation from a separate constant-mean GARCH(1,1) model with Student-t innovations. Values are in percentage-return standard-deviation units on observed market trading dates.

Source: Author's calculations based on market data from Yahoo Finance and Stooq.

## Figure 5. Estimated Association between Lagged Reddit Sentiment and Conditional Market Volatility

- Technical filename: `outputs/figures/sentiment_coefficient_comparison.png`
- Recommended chapter: Chapter 5 — Results
- Optional chapter number: Figure 5.4

Notes: Points show the estimated OLS coefficients on lagged Reddit sentiment (`sentiment_lag1`) from five separate market models. Error bars represent 95% confidence intervals calculated using HAC/Newey-West standard errors with maximum lag 5; the vertical reference line denotes zero. Estimates are associational and should not be interpreted causally.

Source: Author's calculations based on Reddit data, FinBERT sentiment scores, and market data from Yahoo Finance and Stooq.

# Recommended Appendix Outputs

## Appendix tables

### Table A1. Composition of the Final Reddit Sample by Year, Subreddit, and Text Characteristics

- Technical filename: `outputs/tables/reddit_sample_composition.csv`
- Notes: Counts and percentages describe the 1,503 qualifying posts across the three finance-oriented Reddit communities, calendar years, available text, relevance paths, and language-status categories. No balancing or resampling was used.
- Source: Author's calculations based on Reddit data.

### Table A2. Comparison of Post-Weighted and Populated-Day-Weighted Sentiment

- Technical filename: `outputs/tables/sentiment_weighting_comparison.csv`
- Notes: The post-level mean weights each of 1,503 posts equally; the populated-day mean weights each of 504 observed sentiment days equally. The comparison is descriptive and does not define an alternative sentiment index.
- Source: Author's calculations based on Reddit data and FinBERT sentiment scores.

### Table A3. Market Price Data Coverage and Sources

- Technical filename: `outputs/tables/market_price_coverage.csv`
- Notes: Coverage is reported on each index's actual trading dates. Yahoo Finance supplied adjusted closing levels for EURO STOXX 50, DAX, CAC 40, and FTSE 100. WIG20 Close observations are from Stooq, using the preserved archived response documented in the repository.
- Source: Author's calculations based on market data from Yahoo Finance and Stooq; WIG20 uses the preserved archived Stooq response documented in the repository.

### Table A4. Descriptive Statistics of Daily Log Returns

- Technical filename: `outputs/tables/market_return_descriptives.csv`
- Notes: Daily log returns are calculated within each market as the logarithm of the current close level divided by the preceding close level. Decimal returns and their percentage-return transformations are reported for the actual 2021–2023 trading dates.
- Source: Author's calculations based on market data from Yahoo Finance and Stooq.

### Table A5. FinBERT Chunking and Processing Summary

- Technical filename: `outputs/tables/finbert_processing_diagnostics.csv`
- Notes: The table documents conceptual chunks, the 120-chunk cap, tokenizer-safeguard fragments, reconstructed conceptual probability vectors, and final model inputs. Safeguard fragmentation does not change conceptual-chunk weighting in the post-level average.
- Source: Author's calculations based on Reddit data and FinBERT processing diagnostics.

### Table A6. Trading-Day Mapping Reconciliation by Market

- Technical filename: `outputs/tables/trading_day_mapping_reconciliation.csv`
- Notes: Each of the 1,503 post-level FinBERT observations is mapped separately to the same or next actual trading day for each market. The table reconciles same-day and forward mappings and confirms that no post is duplicated or left terminally unmapped.
- Source: Author's calculations based on Reddit data, FinBERT sentiment scores, and market trading calendars.

### Table A7. Trading-Day Reddit Coverage by Market

- Technical filename: `outputs/tables/trading_day_coverage.csv`
- Notes: Coverage is calculated on each market's complete trading calendar after post-level mapping and aggregation. Zero-attention trading days retain missing aligned sentiment; no sentiment value is imputed.
- Source: Author's calculations based on Reddit data, FinBERT sentiment scores, and market trading calendars.

### Table A8. Regression-Sample Retention after Trading-Day Alignment

- Technical filename: `outputs/tables/alignment_sample_sizes.csv`
- Notes: The table traces availability of current volatility and lagged regressors after market-specific alignment. Regression eligibility requires complete approved variables, including observable lagged Reddit sentiment; missing sentiment on prior zero-attention trading days is not imputed.
- Source: Author's calculations.

### Table A9. Support of Lagged Reddit Sentiment and Attention in the Regression Samples

- Technical filename: `outputs/tables/regression_sample_support.csv`
- Notes: The eligible samples contain 397–407 observations per market. Because eligibility requires observed prior-trading-day sentiment, lagged attention is positive in every included row and describes discussion intensity conditional on discussion being observed.
- Source: Author's calculations.

### Table A10. Descriptive Statistics for the Market-Specific Regression Samples

- Technical filename: `outputs/tables/regression_sample_descriptives.csv`
- Notes: Statistics are reported separately for the dependent variable and four approved lagged regressors in each market's complete-case Phase 7 sample. Conditional volatility is in percentage-return standard-deviation units and `return_lag1` is a decimal log return.
- Source: Author's calculations.

### Table A11. Regression Sample Composition by Year

- Technical filename: `outputs/tables/regression_sample_year_composition.csv`
- Notes: Counts and shares describe the year composition of each market-specific Phase 7 estimation sample. The samples are concentrated in 2022 because qualifying Reddit discussion is likewise concentrated in that year.
- Source: Author's calculations.

## Appendix figures

### Figure A1. Distribution of Continuous Post-Level FinBERT Sentiment Scores

- Technical filename: `outputs/figures/finbert_sentiment_score_distribution.png`
- Notes: The histogram shows `sentiment_score`, defined as post-level positive probability minus negative probability, for all 1,503 qualifying posts. Scores range from -1 to 1; the dashed reference line denotes zero.
- Source: Author's calculations based on Reddit data and FinBERT sentiment scores.

### Figure A2. Daily Reddit Attention during 2021–2023

- Technical filename: `outputs/figures/daily_reddit_attention.png`
- Notes: Reddit attention is the number of qualifying posts on each calendar day across the three pooled finance-oriented communities. The complete calendar is shown; zero-post days are recorded as zero attention and no smoothing is applied.
- Source: Author's calculations based on Reddit data.

### Figure A3. Daily Log Returns across the Five European Equity Markets

- Technical filename: `outputs/figures/market_log_returns.png`
- Notes: Panels show untrimmed daily log returns in decimal units on each market's observed trading dates. The dashed horizontal reference line denotes zero; no smoothing, interpolation, or normalization is applied.
- Source: Author's calculations based on market data from Yahoo Finance and Stooq.

### Figure A4. Market-Specific Aligned Reddit Attention

- Technical filename: `outputs/figures/market_aligned_attention.png`
- Notes: Post-level observations are mapped to the same or next actual trading day separately for each market and then counted. Zero-attention trading days remain zero, and aligned sentiment remains missing on those days; no smoothing or interpolation is applied.
- Source: Author's calculations based on Reddit data and market trading calendars.

### Figure A5. Equity-Index Price Levels during 2021–2023

- Technical filename: `outputs/figures/market_price_levels_qc.png`
- Notes: Panels show the original closing levels used to construct returns and use independent vertical scales. Yahoo Finance adjusted closes are used for four indices; WIG20 uses the Stooq Close field from the preserved archived response documented in the repository. The figure is descriptive quality-control support.
- Source: Author's calculations based on market data from Yahoo Finance and Stooq; WIG20 uses the preserved archived Stooq response documented in the repository.

# Technical/QC Outputs Not Recommended for the Thesis Body

These artifacts remain useful for auditability, reproducibility, and researcher review. They should not receive thesis table numbers unless a supervisor specifically requests a technical appendix. Detailed notes and source lines for each item are in the machine-readable catalogue.

## Consolidation and duplicate presentation files

| Human-readable label | Technical filename | Reason for technical-only classification |
|---|---|---|
| Technical QC — Active Script and Rerun Inventory | `outputs/tables/active_script_inventory.csv` | Reproducibility inventory, not an empirical result. |
| Technical QC — Market-Aligned Reddit Descriptive Statistics | `outputs/tables/aligned_reddit_descriptives.csv` | Redundant with selected coverage and support tables. |
| Technical QC — Final Methodological Limitations Synthesis | `outputs/tables/final_methodological_limitations.csv` | Better used as thesis prose. |
| Technical QC — Condensed Phase 8 Regression Display | `outputs/tables/final_regression_table.csv` | Omits p-values and confidence intervals; use the authoritative Table 6 source. |
| Technical QC — Consolidated Final Sample Overview | `outputs/tables/final_sample_overview.csv` | Long-form cross-file review aid. |
| Technical QC — Market-Level Lagged Sentiment Result Summary | `outputs/tables/final_sentiment_results_summary.csv` | Redundant with Table 6 and Figure 5. |
| Technical QC — Final Thesis Output Inventory | `outputs/tables/final_thesis_output_manifest.csv` | Repository inventory, not a thesis result. |
| Technical QC — Long-Form OLS-HAC Coefficient Output | `outputs/tables/regression_coefficients_long.csv` | Machine-friendly duplicate of Table 6 evidence. |
| Technical QC — RQ1, RQ2, and H1 Evidence Synthesis | `outputs/tables/research_question_summary.csv` | Better used in results/discussion prose. |
| Technical QC — Source Data for the Lagged Sentiment Coefficient Figure | `outputs/tables/sentiment_coefficient_comparison.csv` | Source data already presented by Table 6 and Figure 5. |
| Technical QC — Yearly Regression Eligibility and Retention | `outputs/tables/yearly_regression_support.csv` | Superseded for presentation by Table A11. |

## Empirical validation and reproducibility diagnostics

| Human-readable label | Technical filename |
|---|---|
| Technical QC — Trading-Day Alignment Timing Review Sample | `outputs/diagnostics/alignment_timing_review_sample.csv` |
| Technical QC — Post-Level Alignment Weighting Validation | `outputs/diagnostics/alignment_weighting_validation.csv` |
| Technical QC — Extreme Daily Sentiment and Attention Observations | `outputs/diagnostics/daily_reddit_extremes.csv` |
| Technical QC — Daily Reddit Sentiment Construction Summary | `outputs/diagnostics/daily_reddit_sentiment_summary.csv` |
| Technical QC — Descriptive Tables and Figures Validation | `outputs/diagnostics/descriptive_results_validation.csv` |
| Technical QC — Final Frozen-Output Reproducibility Audit | `outputs/diagnostics/final_reproducibility_audit.csv` |
| Technical QC — FinBERT Classification Distribution by Group | `outputs/diagnostics/finbert_class_distribution.csv` |
| Technical QC — FinBERT Development-Only Sample | `outputs/diagnostics/finbert_development_sample.csv` |
| Technical QC — FinBERT Inference Performance Benchmark | `outputs/diagnostics/finbert_inference_benchmark.csv` |
| Technical QC — FinBERT Post-Level Review Sample | `outputs/diagnostics/finbert_review_sample.csv` |
| Technical QC — FinBERT Probability, Chunking, and Sentiment Validation | `outputs/diagnostics/finbert_sentiment_summary.csv` |
| Technical QC — GARCH Model Convergence and Output Validation | `outputs/diagnostics/garch_model_diagnostics.csv` |
| Technical QC — Market Data Coverage and Provenance Summary | `outputs/diagnostics/market_data_summary.csv` |
| Technical QC — Reddit Candidate Corpus Profile | `outputs/diagnostics/reddit_candidate_corpus_profile.csv` |
| Technical QC — Final Reddit Cleaning and Relevance-Filter Summary | `outputs/diagnostics/reddit_cleaning_summary.csv` |
| Technical QC — Combined versus Individual Reddit Query Differences | `outputs/diagnostics/reddit_combined_query_differences.csv` |
| Technical QC — Combined versus Individual Reddit Query Reconciliation | `outputs/diagnostics/reddit_combined_query_validation.csv` |
| Technical QC — Reddit Candidate Extraction Counts | `outputs/diagnostics/reddit_extraction_summary.csv` |
| Technical QC — Manual Review Sample of Excluded Reddit Posts | `outputs/diagnostics/reddit_filter_excluded_review.csv` |
| Technical QC — Manual Review Sample of Included Reddit Posts | `outputs/diagnostics/reddit_filter_included_review.csv` |
| Technical QC — English-Language Classification Review | `outputs/diagnostics/reddit_language_validation.csv` |
| Technical QC — Reddit Extraction Query and Retry Log | `outputs/diagnostics/reddit_query_summary.csv` |
| Technical QC — Historical Reddit Relevance-Filter Dry Run | `outputs/diagnostics/reddit_relevance_filter_dry_run.csv` |
| Technical QC — Candidate-Corpus Relevance Review Sample | `outputs/diagnostics/reddit_relevance_review_sample.csv` |
| Technical QC — Historical Reddit Relevance-Rule Comparison | `outputs/diagnostics/reddit_relevance_rule_comparison.csv` |
| Technical QC — Review of Posts Newly Excluded by the Refined Rule | `outputs/diagnostics/reddit_rule_newly_excluded_review.csv` |
| Technical QC — Review of Posts Newly Included by the Refined Rule | `outputs/diagnostics/reddit_rule_newly_included_review.csv` |
| Technical QC — Full Reddit-to-Market Trading-Day Mapping | `outputs/diagnostics/reddit_trading_day_mapping.csv` |
| Technical QC — OLS-HAC Specification and Input Validation | `outputs/diagnostics/regression_model_validation.csv` |
| Technical QC — Trading-Day Sentiment Missingness Transitions | `outputs/diagnostics/sentiment_missingness_transitions.csv` |
| Technical QC — Trading-Day Mapping, Aggregation, and Lag Validation | `outputs/diagnostics/trading_day_alignment_validation.csv` |
