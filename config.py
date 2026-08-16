"""Central methodology settings and paths for the thesis empirical pipeline."""

from pathlib import Path


# -----------------------------------------------------------------------------
# Sample
# -----------------------------------------------------------------------------

START_DATE = "2021-01-01"
END_DATE = "2023-12-31"


# -----------------------------------------------------------------------------
# Reddit sources
# -----------------------------------------------------------------------------

SUBREDDITS = [
    "investing",
    "stocks",
    "StockMarket",
]

# Broad candidate-retrieval terms for Phase 3A. These are extraction keywords,
# not the final Ukraine-war financial relevance rule used in Phase 3B.
REDDIT_EXTRACTION_KEYWORDS = [
    "Ukraine",
    "Ukrainian",
    "Russia",
    "Russian",
    "Putin",
    "Kyiv",
    "Kiev",
    "Moscow",
    "invasion",
    "sanctions",
    "NATO",
]

# Transparent Phase 3B.2 candidate-relevance vocabularies. These support a
# dry-run rule that must be manually validated before it becomes final.
REDDIT_DIRECT_UKRAINE_CONTEXT_TERMS = [
    "Ukraine",
    "Ukrainian",
    "Kyiv",
    "Kiev",
    "Crimea",
    "Donbas",
    "Donetsk",
    "Luhansk",
    "Zelensky",
    "Zelenskiy",
]

REDDIT_RUSSIA_CONTEXT_TERMS = [
    "Russia",
    "Russian",
    "Putin",
    "Moscow",
    "Kremlin",
]

REDDIT_CONFLICT_CONTEXT_TERMS = [
    "war",
    "wars",
    "invasion",
    "invasions",
    "invade",
    "invades",
    "invaded",
    "invading",
    "conflict",
    "conflicts",
    "troop",
    "troops",
    "military",
    "sanction",
    "sanctions",
    "sanctioned",
    "sanctioning",
    "NATO",
    "embargo",
    "embargoes",
    "embargoed",
    "mobilization",
    "mobilisation",
    "annexation",
    "ceasefire",
    "cease-fire",
]

REDDIT_FINANCIAL_CONTEXT_TERMS = [
    "stock",
    "stocks",
    "market",
    "markets",
    "equity",
    "equities",
    "index",
    "indices",
    "investor",
    "investors",
    "investing",
    "investment",
    "trading",
    "portfolio",
    "share",
    "shares",
    "bond",
    "bonds",
    "yield",
    "yields",
    "ETF",
    "ETFs",
    "ADR",
    "ADRs",
    "bank",
    "banks",
    "banking",
    "finance",
    "financial",
    "economy",
    "economic",
    "recession",
    "inflation",
    "interest rate",
    "interest rates",
    "currency",
    "currencies",
    "ruble",
    "rouble",
    "oil",
    "gas",
    "energy",
    "commodity",
    "commodities",
    "wheat",
    "metals",
    "gold",
    "fund",
    "funds",
    "asset",
    "assets",
    "securities",
    "debt",
    "default",
    "earnings",
    "company",
    "companies",
    "price",
    "prices",
    "volatility",
    "risk",
    "futures",
    "options",
]

REDDIT_CRISIS_FINANCIAL_CONSEQUENCE_PHRASES = [
    "Russian default",
    "Russia default",
    "sovereign default",
    "capital controls",
    "frozen assets",
    "frozen securities",
    "blocked securities",
    "SWIFT",
    "remove from SWIFT",
    "removed from SWIFT",
    "exclude from SWIFT",
    "divest from Russia",
    "exit Russia",
    "exiting Russia",
    "leave Russia",
    "leaving Russia",
    "Russian assets",
    "Russian debt",
    "Russian bonds",
    "Russian ADRs",
    "ruble collapse",
    "rouble collapse",
    "oil embargo",
    "energy embargo",
]

# Phase 3B.3 refinement. Keep the Phase 3B.2 lists above unchanged so the
# previous dry run remains reproducible and auditable.
REDDIT_REFINED_CONFLICT_CONTEXT_TERMS = [
    *REDDIT_CONFLICT_CONTEXT_TERMS,
    "tension",
    "tensions",
    "escalation",
    "escalating",
    "escalate",
    "attack",
    "attacks",
    "attacked",
    "threat",
    "threats",
    "troop buildup",
    "troop build-up",
    "border buildup",
    "border build-up",
    "offensive",
    "crisis",
]

REDDIT_REFINED_FINANCIAL_CONTEXT_TERMS = [
    *REDDIT_FINANCIAL_CONTEXT_TERMS,
    "hedge",
    "hedging",
    "profit",
    "profits",
    "profitable",
    "money",
    "exposure",
    "loss",
    "losses",
    "gain",
    "gains",
    "return",
    "returns",
    "buy",
    "buying",
    "sell",
    "selling",
    "short",
    "shorting",
]

REDDIT_REFINED_CRISIS_FINANCIAL_CONSEQUENCE_PHRASES = [
    "frozen assets",
    "assets frozen",
    "frozen securities",
    "blocked securities",
    "asset freeze",
    "capital controls",
    "removed from SWIFT",
    "remove from SWIFT",
    "excluded from SWIFT",
    "exclude from SWIFT",
    "SWIFT sanctions",
    "divest from Russia",
    "exit Russia",
    "exiting Russia",
    "leave Russia",
    "leaving Russia",
    "oil embargo",
    "energy embargo",
]

# Final Phase 3B rule. These lists freeze the validated Phase 3B.3 vocabulary
# and add only the narrow variants observed during manual review.
REDDIT_FINAL_CONFLICT_CONTEXT_TERMS = [
    *REDDIT_REFINED_CONFLICT_CONTEXT_TERMS,
    "crises",
]

REDDIT_FINAL_CRISIS_FINANCIAL_CONSEQUENCE_PHRASES = [
    *REDDIT_REFINED_CRISIS_FINANCIAL_CONSEQUENCE_PHRASES,
    "cut off from SWIFT",
    "cut Russia off from SWIFT",
    "cut Russian banks off from SWIFT",
    "cut some Russian banks off from SWIFT",
    "kicked off SWIFT",
    "kicked Russia off SWIFT",
    "kicked out of SWIFT",
    "kicked Russia out of SWIFT",
    "taken off SWIFT",
    "taken off of SWIFT",
    "trading halt",
    "trading halted",
    "trading suspension",
    "suspended trading",
    "ADR suspension",
    "ADR conversion",
    "ADR delisting",
    "delisted ADR",
    "Russian ADRs suspended",
    "Russian ADRs have been suspended",
    "suspended Russian ADRs",
    "delisting process for Russian ADRs",
    "Russian securities suspended",
    "Russian securities frozen",
    "freeze funds exposed to Russia",
    "freeze funds exposed to Russian assets",
]

REDDIT_LANGUAGE_DETECTOR_SEED = 0
REDDIT_LANGUAGE_MIN_ALPHA_WORDS = 8
REDDIT_LANGUAGE_MIN_ALPHA_CHARACTERS = 40
REDDIT_LANGUAGE_CONFIDENT_ENGLISH_PROBABILITY = 0.80
REDDIT_LANGUAGE_CONFIDENT_NON_ENGLISH_PROBABILITY = 0.99
REDDIT_LANGUAGE_CONFIDENT_NON_ENGLISH_MIN_WORDS = 20
REDDIT_MAX_EXPECTED_FINAL_RELEVANCE_INCREASE = 100


# -----------------------------------------------------------------------------
# FinBERT
# -----------------------------------------------------------------------------

FINBERT_MODEL = "ProsusAI/finbert"
CHUNK_MAX_WORDS = 30
MAX_CHUNKS_PER_POST = 120
FINBERT_BATCH_SIZE = 8
FINBERT_CHECKPOINT_INTERVAL = 800


# -----------------------------------------------------------------------------
# GARCH volatility
# -----------------------------------------------------------------------------

GARCH_MEAN_MODEL = "Constant"
GARCH_VOLATILITY_MODEL = "GARCH"
GARCH_P = 1
GARCH_O = 0
GARCH_Q = 1
GARCH_DISTRIBUTION = "StudentsT"


# -----------------------------------------------------------------------------
# Regression
# -----------------------------------------------------------------------------

HAC_MAX_LAGS = 5


# -----------------------------------------------------------------------------
# Equity markets
# -----------------------------------------------------------------------------

YAHOO_MARKET_TICKERS = {
    "EURO_STOXX_50": "^STOXX50E",
    "DAX": "^GDAXI",
    "CAC_40": "^FCHI",
    "FTSE_100": "^FTSE",
}

STOOQ_WIG20_SYMBOL = "wig20"
STOOQ_DOWNLOAD_URL = "https://stooq.pl/q/d/l/"

# Stooq introduced an API key and browser verification after the thesis sample
# was collected. This capture is a preserved response from Stooq's own WIG20
# CSV endpoint and is used only when the live endpoint does not return CSV.
STOOQ_WIG20_ARCHIVE_URL = (
    "https://web.archive.org/web/20250114102640id_/"
    "https://stooq.com/q/d/l/?s=wig20&i=d"
)

MARKET_TICKERS = {
    **YAHOO_MARKET_TICKERS,
    "WIG20": "WIG20",
}

MARKET_DATA_SOURCES = {
    **{index_name: "Yahoo Finance" for index_name in YAHOO_MARKET_TICKERS},
    "WIG20": "Stooq",
}

WIG20_YEAR_END_LEVELS = {
    2021: 2266.92,
    2022: 1792.01,
    2023: 2342.99,
}
WIG20_YEAR_END_RELATIVE_TOLERANCE = 0.05


# -----------------------------------------------------------------------------
# Repository paths
# -----------------------------------------------------------------------------

REPOSITORY_ROOT = Path(__file__).resolve().parent

DATA_DIR = REPOSITORY_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
RAW_MARKET_DATA_DIR = RAW_DATA_DIR / "market"
RAW_REDDIT_DATA_DIR = RAW_DATA_DIR / "reddit"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

SRC_DIR = REPOSITORY_ROOT / "src"

OUTPUTS_DIR = REPOSITORY_ROOT / "outputs"
TABLES_DIR = OUTPUTS_DIR / "tables"
FIGURES_DIR = OUTPUTS_DIR / "figures"
DIAGNOSTICS_DIR = OUTPUTS_DIR / "diagnostics"

ARCHIVE_DIR = REPOSITORY_ROOT / "archive"

RAW_REDDIT_FILE = RAW_REDDIT_DATA_DIR / "reddit_posts_2021_2023_raw.csv"
REDDIT_CHECKPOINT_DIR = RAW_REDDIT_DATA_DIR / "checkpoints"
REDDIT_CHECKPOINT_POSTS_FILE = REDDIT_CHECKPOINT_DIR / "reddit_posts_checkpoint.csv"
REDDIT_CHECKPOINT_WINDOWS_FILE = (
    REDDIT_CHECKPOINT_DIR / "reddit_windows_checkpoint.csv"
)
REDDIT_EXTRACTION_SUMMARY_FILE = (
    DIAGNOSTICS_DIR / "reddit_extraction_summary.csv"
)
REDDIT_QUERY_SUMMARY_FILE = DIAGNOSTICS_DIR / "reddit_query_summary.csv"
REDDIT_RELEVANCE_REVIEW_SAMPLE_FILE = (
    DIAGNOSTICS_DIR / "reddit_relevance_review_sample.csv"
)
REDDIT_CANDIDATE_CORPUS_PROFILE_FILE = (
    DIAGNOSTICS_DIR / "reddit_candidate_corpus_profile.csv"
)
REDDIT_REVIEW_RANDOM_SEED = 20250301
REDDIT_RELEVANCE_FILTER_DRY_RUN_FILE = (
    DIAGNOSTICS_DIR / "reddit_relevance_filter_dry_run.csv"
)
REDDIT_FILTER_INCLUDED_REVIEW_FILE = (
    DIAGNOSTICS_DIR / "reddit_filter_included_review.csv"
)
REDDIT_FILTER_EXCLUDED_REVIEW_FILE = (
    DIAGNOSTICS_DIR / "reddit_filter_excluded_review.csv"
)
REDDIT_FILTER_REVIEW_RANDOM_SEED = 20250302
REDDIT_RELEVANCE_RULE_COMPARISON_FILE = (
    DIAGNOSTICS_DIR / "reddit_relevance_rule_comparison.csv"
)
REDDIT_RULE_NEWLY_INCLUDED_REVIEW_FILE = (
    DIAGNOSTICS_DIR / "reddit_rule_newly_included_review.csv"
)
REDDIT_RULE_NEWLY_EXCLUDED_REVIEW_FILE = (
    DIAGNOSTICS_DIR / "reddit_rule_newly_excluded_review.csv"
)
REDDIT_RULE_COMPARISON_RANDOM_SEED = 20250303
CLEANED_REDDIT_FILE = PROCESSED_DATA_DIR / "reddit_posts_cleaned.csv"
FINBERT_REDDIT_FILE = PROCESSED_DATA_DIR / "reddit_posts_finbert.csv"
DAILY_REDDIT_FILE = PROCESSED_DATA_DIR / "daily_reddit_sentiment.csv"
REDDIT_CLEANING_SUMMARY_FILE = (
    DIAGNOSTICS_DIR / "reddit_cleaning_summary.csv"
)
REDDIT_LANGUAGE_VALIDATION_FILE = (
    DIAGNOSTICS_DIR / "reddit_language_validation.csv"
)
FINBERT_SENTIMENT_SUMMARY_FILE = (
    DIAGNOSTICS_DIR / "finbert_sentiment_summary.csv"
)
FINBERT_CLASS_DISTRIBUTION_FILE = (
    DIAGNOSTICS_DIR / "finbert_class_distribution.csv"
)
FINBERT_REVIEW_SAMPLE_FILE = (
    DIAGNOSTICS_DIR / "finbert_review_sample.csv"
)
FINBERT_DEVELOPMENT_SAMPLE_FILE = (
    DIAGNOSTICS_DIR / "finbert_development_sample.csv"
)
FINBERT_BENCHMARK_FILE = DIAGNOSTICS_DIR / "finbert_inference_benchmark.csv"
DAILY_REDDIT_SUMMARY_FILE = (
    DIAGNOSTICS_DIR / "daily_reddit_sentiment_summary.csv"
)

# Retrospective descriptive results for validated Phases 2 through 4B.
REDDIT_SAMPLE_CONSTRUCTION_TABLE = (
    TABLES_DIR / "reddit_sample_construction.csv"
)
REDDIT_SAMPLE_COMPOSITION_TABLE = TABLES_DIR / "reddit_sample_composition.csv"
FINBERT_POST_DESCRIPTIVES_TABLE = (
    TABLES_DIR / "finbert_post_sentiment_descriptives.csv"
)
FINBERT_PROCESSING_DIAGNOSTICS_TABLE = (
    TABLES_DIR / "finbert_processing_diagnostics.csv"
)
DAILY_REDDIT_DESCRIPTIVES_TABLE = (
    TABLES_DIR / "daily_reddit_descriptives.csv"
)
SENTIMENT_WEIGHTING_COMPARISON_TABLE = (
    TABLES_DIR / "sentiment_weighting_comparison.csv"
)
MARKET_PRICE_COVERAGE_TABLE = TABLES_DIR / "market_price_coverage.csv"
DAILY_REDDIT_EXTREMES_FILE = DIAGNOSTICS_DIR / "daily_reddit_extremes.csv"
DESCRIPTIVE_RESULTS_VALIDATION_FILE = (
    DIAGNOSTICS_DIR / "descriptive_results_validation.csv"
)
REDDIT_SAMPLE_COMPOSITION_FIGURE = (
    FIGURES_DIR / "reddit_sample_composition.png"
)
FINBERT_SENTIMENT_SCORE_FIGURE = (
    FIGURES_DIR / "finbert_sentiment_score_distribution.png"
)
FINBERT_LABEL_DISTRIBUTION_FIGURE = (
    FIGURES_DIR / "finbert_label_distribution.png"
)
DAILY_REDDIT_SENTIMENT_FIGURE = FIGURES_DIR / "daily_reddit_sentiment.png"
DAILY_REDDIT_ATTENTION_FIGURE = FIGURES_DIR / "daily_reddit_attention.png"
MARKET_PRICE_LEVELS_QC_FIGURE = FIGURES_DIR / "market_price_levels_qc.png"

# Phase 5 market returns and GARCH(1,1)-Student-t volatility.
MARKET_RETURNS_GARCH_FILE = PROCESSED_DATA_DIR / "market_returns_garch.csv"
MARKET_RETURN_DESCRIPTIVES_TABLE = (
    TABLES_DIR / "market_return_descriptives.csv"
)
GARCH_PARAMETERS_TABLE = TABLES_DIR / "garch_parameters.csv"
GARCH_VOLATILITY_DESCRIPTIVES_TABLE = (
    TABLES_DIR / "garch_volatility_descriptives.csv"
)
GARCH_MODEL_DIAGNOSTICS_FILE = (
    DIAGNOSTICS_DIR / "garch_model_diagnostics.csv"
)
MARKET_LOG_RETURNS_FIGURE = FIGURES_DIR / "market_log_returns.png"
GARCH_CONDITIONAL_VOLATILITY_FIGURE = (
    FIGURES_DIR / "garch_conditional_volatility.png"
)

FINBERT_CHECKPOINT_DIR = REPOSITORY_ROOT / "checkpoints" / "finbert"
MARKET_PRICES_FILE = PROCESSED_DATA_DIR / "market_prices.csv"
ANALYSIS_DATASET_FILE = PROCESSED_DATA_DIR / "analysis_dataset.csv"
MARKET_DATA_SUMMARY_FILE = DIAGNOSTICS_DIR / "market_data_summary.csv"

MARKET_PRICE_FILES = {
    "EURO_STOXX_50": RAW_MARKET_DATA_DIR / "euro_stoxx_50_close_level.csv",
    "DAX": RAW_MARKET_DATA_DIR / "dax_close_level.csv",
    "CAC_40": RAW_MARKET_DATA_DIR / "cac_40_close_level.csv",
    "FTSE_100": RAW_MARKET_DATA_DIR / "ftse_100_close_level.csv",
    "WIG20": RAW_MARKET_DATA_DIR / "wig20_close_level.csv",
}
