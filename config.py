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
