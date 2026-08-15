"""Central configuration for the thesis empirical pipeline.

Phase 1 defines methodology constants and repository paths only. Empirical
processing logic is implemented in later phases.
"""

from pathlib import Path


# -----------------------------------------------------------------------------
# Sample
# -----------------------------------------------------------------------------

START_DATE = "2021-01-01"
END_DATE = "2023-12-31"


# -----------------------------------------------------------------------------
# Reddit sources
# -----------------------------------------------------------------------------

SUBREDDITS = ["investing", "stocks"]

# Reddit relevance-filter configuration (reserved for the Reddit phase)
# The final Ukraine-war financial relevance rules are deliberately not defined
# here yet. Generic energy, inflation, or recession terms will not be sufficient
# by themselves when those rules are implemented.


# -----------------------------------------------------------------------------
# FinBERT
# -----------------------------------------------------------------------------

FINBERT_MODEL = "ProsusAI/finbert"
CHUNK_MAX_WORDS = 30
MAX_CHUNKS_PER_POST = 120


# -----------------------------------------------------------------------------
# Regression
# -----------------------------------------------------------------------------

HAC_MAX_LAGS = 5


# -----------------------------------------------------------------------------
# Equity markets
# -----------------------------------------------------------------------------

MARKET_TICKERS = {
    "EURO_STOXX_50": "^STOXX50E",
    "DAX": "^GDAXI",
    "CAC_40": "^FCHI",
    "FTSE_100": "^FTSE",
    "WIG20": "WIG20.WA",
}


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
ANALYSIS_DATASET_FILE = PROCESSED_DATA_DIR / "analysis_dataset.csv"

MARKET_PRICE_FILES = {
    "EURO_STOXX_50": RAW_MARKET_DATA_DIR / "euro_stoxx_50_adjusted_close.csv",
    "DAX": RAW_MARKET_DATA_DIR / "dax_adjusted_close.csv",
    "CAC_40": RAW_MARKET_DATA_DIR / "cac_40_adjusted_close.csv",
    "FTSE_100": RAW_MARKET_DATA_DIR / "ftse_100_adjusted_close.csv",
    "WIG20": RAW_MARKET_DATA_DIR / "wig20_adjusted_close.csv",
}

