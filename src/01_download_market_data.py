"""Phase 2: collect and validate the five European equity-index series.

Why this phase exists
    The return and GARCH stages require one trustworthy closing-level series
    for each market over the common 2021-2023 thesis period.

Main inputs
    Remote Yahoo Finance data for EURO STOXX 50, DAX, CAC 40, and FTSE 100;
    Stooq symbol ``wig20`` for WIG20, with the validated archived January 2025
    Stooq response as the documented fallback.

Main outputs
    Source-specific CSV files, ``data/processed/market_prices.csv``, and the
    market-data provenance/quality summary under ``outputs/diagnostics``.

Methodological rules and boundaries
    Yahoo adjusted closes and the Stooq daily Close field are standardized as
    ``close_level``. The script checks positive prices, unique market dates,
    source identity, and sample coverage. It does not calculate returns,
    estimate GARCH models, or use WIG20.WA/GPW.WA as a proxy for WIG20.
"""

from __future__ import annotations

import os
import sys
from datetime import date, timedelta
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf


# Allow ``python src/01_download_market_data.py`` from the repository root.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from config import (  # noqa: E402
    END_DATE,
    MARKET_DATA_SOURCES,
    MARKET_DATA_SUMMARY_FILE,
    MARKET_PRICE_FILES,
    MARKET_PRICES_FILE,
    MARKET_TICKERS,
    START_DATE,
    STOOQ_DOWNLOAD_URL,
    STOOQ_WIG20_ARCHIVE_URL,
    STOOQ_WIG20_SYMBOL,
    WIG20_YEAR_END_LEVELS,
    WIG20_YEAR_END_RELATIVE_TOLERANCE,
    YAHOO_MARKET_TICKERS,
)


CORE_COLUMNS = ["date", "index_name", "ticker", "data_source", "close_level"]
SUMMARY_COLUMNS = [
    "index_name",
    "ticker",
    "data_source",
    "retrieval_method",
    "source_reference",
    "number_of_observations",
    "first_date",
    "last_date",
    "missing_close_level_values",
    "duplicate_dates",
]
REQUEST_TIMEOUT_SECONDS = 60
STOOQ_API_KEY_ENVIRONMENT_VARIABLE = "STOOQ_API_KEY"


# ---------------------------------------------------------------------------
# Validate sources and standardize all prices to the common schema
# ---------------------------------------------------------------------------

def exclusive_download_end(end_date: str) -> str:
    """Return the day after the inclusive thesis end date for yfinance."""

    return (date.fromisoformat(end_date) + timedelta(days=1)).isoformat()


def validate_configuration() -> None:
    """Check that market sources and output paths are configured consistently."""

    expected_indices = set(MARKET_TICKERS)
    if expected_indices != set(MARKET_PRICE_FILES):
        raise ValueError(
            "MARKET_TICKERS and MARKET_PRICE_FILES must contain the same indices."
        )
    if expected_indices != set(MARKET_DATA_SOURCES):
        raise ValueError(
            "MARKET_TICKERS and MARKET_DATA_SOURCES must contain the same indices."
        )
    if len(expected_indices) != 5:
        raise ValueError("Exactly five thesis market indices must be configured.")
    if set(YAHOO_MARKET_TICKERS) != expected_indices - {"WIG20"}:
        raise ValueError("Only the four non-WIG20 indices may use Yahoo Finance.")
    if MARKET_TICKERS.get("WIG20") != "WIG20":
        raise ValueError("Stooq WIG20 observations must use ticker WIG20.")
    if STOOQ_WIG20_SYMBOL.lower() != "wig20":
        raise ValueError("The Stooq WIG20 download symbol must be wig20.")
    if any("GPW.WA" in ticker for ticker in MARKET_TICKERS.values()):
        raise ValueError("GPW.WA is not WIG20 and must not be used.")


def extract_adjusted_close(downloaded: pd.DataFrame, ticker: str) -> pd.Series:
    """Extract one adjusted-close series from a yfinance response."""

    if downloaded.empty:
        raise ValueError(f"No data returned for {ticker}.")

    if isinstance(downloaded.columns, pd.MultiIndex):
        exact_matches = [
            column
            for column in downloaded.columns
            if "Adj Close" in column and ticker in column
        ]
        matches = exact_matches or [
            column for column in downloaded.columns if "Adj Close" in column
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected one adjusted-close column for {ticker}, found {len(matches)}."
            )
        adjusted_close = downloaded[matches[0]]
    else:
        if "Adj Close" not in downloaded.columns:
            raise ValueError(f"Adjusted-close column is missing for {ticker}.")
        adjusted_close = downloaded["Adj Close"]

    if isinstance(adjusted_close, pd.DataFrame):
        if adjusted_close.shape[1] != 1:
            raise ValueError(f"Adjusted-close data are ambiguous for {ticker}.")
        adjusted_close = adjusted_close.iloc[:, 0]

    return adjusted_close


def standardize_yahoo_data(
    downloaded: pd.DataFrame,
    index_name: str,
    ticker: str,
) -> pd.DataFrame:
    """Store Yahoo Finance adjusted closes in the common close-level schema."""

    adjusted_close = extract_adjusted_close(downloaded, ticker)
    parsed_dates = pd.to_datetime(downloaded.index, errors="coerce", utc=True)
    parsed_dates = parsed_dates.tz_convert(None).normalize()

    standardized = pd.DataFrame(
        {
            "date": parsed_dates,
            "index_name": index_name,
            "ticker": ticker,
            "data_source": "Yahoo Finance",
            "close_level": pd.to_numeric(
                adjusted_close.to_numpy(), errors="coerce"
            ),
        }
    )
    return standardized.loc[:, CORE_COLUMNS].sort_values("date").reset_index(drop=True)


def parse_stooq_csv(csv_text: str, source_url: str) -> pd.DataFrame:
    """Parse a response only when it contains genuine Stooq price columns."""

    stripped_text = csv_text.lstrip()
    if not stripped_text or stripped_text.lower().startswith(("<!doctype", "<html")):
        raise ValueError(f"Stooq returned HTML instead of price data: {source_url}")

    try:
        downloaded = pd.read_csv(StringIO(csv_text))
    except Exception as error:
        raise ValueError(f"Stooq response is not valid CSV: {source_url}") from error

    required_columns = {"Date", "Close"}
    if downloaded.empty or not required_columns.issubset(downloaded.columns):
        raise ValueError(
            "Stooq response does not contain non-empty Date and Close columns: "
            f"{source_url}"
        )
    return downloaded


def standardize_stooq_data(downloaded: pd.DataFrame) -> pd.DataFrame:
    """Store Stooq WIG20 Close values in the common close-level schema."""

    standardized = pd.DataFrame(
        {
            "date": pd.to_datetime(downloaded["Date"], errors="coerce"),
            "index_name": "WIG20",
            "ticker": MARKET_TICKERS["WIG20"],
            "data_source": "Stooq",
            "close_level": pd.to_numeric(downloaded["Close"], errors="coerce"),
        }
    )
    sample_start = pd.Timestamp(START_DATE)
    sample_end = pd.Timestamp(END_DATE)
    standardized = standardized.loc[
        standardized["date"].between(sample_start, sample_end, inclusive="both")
    ]
    return standardized.loc[:, CORE_COLUMNS].sort_values("date").reset_index(drop=True)


def validate_index_data(
    data: pd.DataFrame,
    index_name: str,
    ticker: str,
    data_source: str,
) -> None:
    """Validate one standardized market-price series.

    Unique, ordered dates prevent duplicated trading observations; positive
    finite close levels are required before logarithmic returns can be formed.
    """

    if data.empty:
        raise ValueError(f"{index_name} ({ticker}) contains no observations.")
    if list(data.columns) != CORE_COLUMNS:
        raise ValueError(f"{index_name} does not have the required columns.")
    if data["date"].isna().any():
        raise ValueError(f"{index_name} contains invalid dates.")
    if not data["date"].is_monotonic_increasing:
        raise ValueError(f"{index_name} dates are not sorted ascending.")

    duplicate_dates = int(data["date"].duplicated().sum())
    if duplicate_dates:
        raise ValueError(f"{index_name} contains {duplicate_dates} duplicate dates.")
    if data["close_level"].isna().any():
        raise ValueError(f"{index_name} contains missing close-level values.")

    close_levels = data["close_level"].to_numpy(dtype=float)
    if not np.isfinite(close_levels).all():
        raise ValueError(f"{index_name} contains non-finite close-level values.")
    if (close_levels <= 0).any():
        raise ValueError(f"{index_name} contains non-positive close-level values.")

    sample_start = pd.Timestamp(START_DATE)
    sample_end = pd.Timestamp(END_DATE)
    outside_sample = ~data["date"].between(sample_start, sample_end, inclusive="both")
    if outside_sample.any():
        raise ValueError(f"{index_name} contains dates outside the thesis period.")
    if not data["index_name"].eq(index_name).all():
        raise ValueError(f"{index_name} contains inconsistent index names.")
    if not data["ticker"].eq(ticker).all():
        raise ValueError(f"{index_name} contains inconsistent ticker values.")
    if not data["data_source"].eq(data_source).all():
        raise ValueError(f"{index_name} contains inconsistent data sources.")


def validate_wig20_year_end_levels(data: pd.DataFrame) -> pd.DataFrame:
    """Compare each WIG20 year-end observation with the supplied benchmarks."""

    comparisons = []
    for year, expected_level in WIG20_YEAR_END_LEVELS.items():
        year_data = data.loc[data["date"].dt.year == year]
        if year_data.empty:
            raise ValueError(f"WIG20 has no observations for {year}.")

        final_row = year_data.iloc[-1]
        actual_level = float(final_row["close_level"])
        relative_difference = abs(actual_level - expected_level) / expected_level
        if relative_difference > WIG20_YEAR_END_RELATIVE_TOLERANCE:
            raise ValueError(
                f"WIG20 {year} final close {actual_level:.2f} materially differs "
                f"from benchmark {expected_level:.2f} "
                f"({relative_difference:.2%} difference)."
            )
        comparisons.append(
            {
                "year": year,
                "last_trading_date": final_row["date"].date().isoformat(),
                "close_level": actual_level,
                "benchmark": expected_level,
                "relative_difference": relative_difference,
            }
        )
    return pd.DataFrame(comparisons)


def download_yahoo_market(index_name: str, ticker: str) -> pd.DataFrame:
    """Download and validate one Yahoo Finance market index."""

    downloaded = yf.download(
        ticker,
        start=START_DATE,
        end=exclusive_download_end(END_DATE),
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    standardized = standardize_yahoo_data(downloaded, index_name, ticker)
    validate_index_data(standardized, index_name, ticker, "Yahoo Finance")
    return standardized


def request_stooq_csv(url: str, params: dict[str, str] | None = None) -> pd.DataFrame:
    """Request a Stooq CSV and reject HTTP-success responses that are not data."""

    response = requests.get(
        url,
        params=params,
        headers={"User-Agent": "Master-thesis market-data pipeline"},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return parse_stooq_csv(response.text, response.url)


def download_stooq_wig20() -> tuple[pd.DataFrame, str | None]:
    """Download WIG20 from Stooq, with a preserved Stooq response as fallback."""

    params = {
        "s": STOOQ_WIG20_SYMBOL,
        "d1": START_DATE.replace("-", ""),
        "d2": END_DATE.replace("-", ""),
        "i": "d",
    }
    api_key = os.environ.get(STOOQ_API_KEY_ENVIRONMENT_VARIABLE)
    if api_key:
        params["apikey"] = api_key

    # The archived response is still Stooq-produced data. The fallback keeps
    # that provenance explicit when the live endpoint returns browser HTML.
    fallback_warning = None
    try:
        downloaded = request_stooq_csv(STOOQ_DOWNLOAD_URL, params=params)
    except Exception as live_error:
        fallback_warning = (
            "Live Stooq CSV unavailable or invalid; used the preserved "
            f"2025-01-14 response from Stooq's own wig20 endpoint. ({live_error})"
        )
        downloaded = request_stooq_csv(STOOQ_WIG20_ARCHIVE_URL)

    standardized = standardize_stooq_data(downloaded)
    validate_index_data(standardized, "WIG20", MARKET_TICKERS["WIG20"], "Stooq")
    validate_wig20_year_end_levels(standardized)
    return standardized, fallback_warning


def build_quality_summary(market_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build the requested market-level quality-control summary."""

    rows = []
    for index_name, data in market_data.items():
        if index_name == "WIG20":
            retrieval_method = "Internet Archive snapshot of Stooq"
            source_reference = STOOQ_WIG20_ARCHIVE_URL
        else:
            retrieval_method = "Yahoo Finance API"
            source_reference = ""

        rows.append(
            {
                "index_name": index_name,
                "ticker": data["ticker"].iloc[0],
                "data_source": data["data_source"].iloc[0],
                "retrieval_method": retrieval_method,
                "source_reference": source_reference,
                "number_of_observations": len(data),
                "first_date": data["date"].min().date().isoformat(),
                "last_date": data["date"].max().date().isoformat(),
                "missing_close_level_values": int(data["close_level"].isna().sum()),
                "duplicate_dates": int(data["date"].duplicated().sum()),
            }
        )
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


# ---------------------------------------------------------------------------
# Reconcile the five-market panel and write canonical Phase 2 outputs
# ---------------------------------------------------------------------------

def validate_long_market_data(data: pd.DataFrame) -> None:
    """Validate the combined five-market long dataset."""

    if data.empty:
        raise ValueError("The combined market-price dataset is empty.")
    if list(data.columns) != CORE_COLUMNS:
        raise ValueError("The combined market-price dataset has incorrect columns.")

    duplicate_pairs = int(data.duplicated(["index_name", "date"]).sum())
    if duplicate_pairs:
        raise ValueError(
            "The combined market-price dataset contains "
            f"{duplicate_pairs} duplicate (index_name, date) pairs."
        )
    if set(data["index_name"].unique()) != set(MARKET_TICKERS):
        raise ValueError("The combined dataset does not contain all five indices.")
    if data["ticker"].astype(str).str.contains("GPW.WA", regex=False).any():
        raise ValueError("GPW.WA appears in the active market dataset.")

    for index_name, ticker in MARKET_TICKERS.items():
        index_data = data.loc[data["index_name"] == index_name].copy()
        validate_index_data(
            index_data,
            index_name,
            ticker,
            MARKET_DATA_SOURCES[index_name],
        )


def save_outputs(
    market_data: dict[str, pd.DataFrame],
    combined_data: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    """Save raw, processed, and diagnostic market-data outputs."""

    for output_file in MARKET_PRICE_FILES.values():
        output_file.parent.mkdir(parents=True, exist_ok=True)
    MARKET_PRICES_FILE.parent.mkdir(parents=True, exist_ok=True)
    MARKET_DATA_SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)

    for index_name, data in market_data.items():
        output_file = MARKET_PRICE_FILES[index_name]
        data.to_csv(output_file, index=False, date_format="%Y-%m-%d")
        print(f"Saved {index_name} raw data to {output_file}")

    combined_data.to_csv(MARKET_PRICES_FILE, index=False, date_format="%Y-%m-%d")
    summary.to_csv(MARKET_DATA_SUMMARY_FILE, index=False)
    print(f"Saved combined market data to {MARKET_PRICES_FILE}")
    print(f"Saved quality-control summary to {MARKET_DATA_SUMMARY_FILE}")


def main() -> None:
    """Run the complete closing-level market-data pipeline."""

    validate_configuration()
    print(
        "Downloading market closing levels for "
        f"{START_DATE} through {END_DATE}. Yahoo Finance uses adjusted closes; "
        "Stooq WIG20 uses Close."
    )

    market_data: dict[str, pd.DataFrame] = {}
    failures: list[tuple[str, str, str]] = []
    warnings: list[str] = []

    for index_name, ticker in YAHOO_MARKET_TICKERS.items():
        print(f"Downloading {index_name} ({ticker}) from Yahoo Finance...")
        try:
            market_data[index_name] = download_yahoo_market(index_name, ticker)
        except Exception as error:
            failures.append((index_name, ticker, str(error)))
            print(f"ERROR: {index_name} ({ticker}) failed: {error}")

    print(f"Downloading WIG20 ({STOOQ_WIG20_SYMBOL}) from Stooq...")
    try:
        market_data["WIG20"], fallback_warning = download_stooq_wig20()
        if fallback_warning:
            warnings.append(fallback_warning)
            print(f"WARNING: {fallback_warning}")
    except Exception as error:
        failures.append(("WIG20", STOOQ_WIG20_SYMBOL, str(error)))
        print(f"ERROR: WIG20 ({STOOQ_WIG20_SYMBOL}) failed: {error}")

    for index_name, data in market_data.items():
        print(
            f"Validated {index_name}: {len(data)} observations, "
            f"{data['date'].min().date().isoformat()} through "
            f"{data['date'].max().date().isoformat()}."
        )

    if failures:
        print("\nMarket-data pipeline incomplete. Failed markets:")
        for index_name, ticker, error in failures:
            print(f"- {index_name} ({ticker}): {error}")
        print("No output files were created or overwritten during this run.")
        raise RuntimeError(f"Market-data download failed for {len(failures)} market(s).")

    combined_data = pd.concat(market_data.values(), ignore_index=True)
    combined_data = combined_data.loc[:, CORE_COLUMNS]
    validate_long_market_data(combined_data)

    wig20_year_ends = validate_wig20_year_end_levels(market_data["WIG20"])
    print("\nWIG20 year-end sanity checks:")
    print(wig20_year_ends.to_string(index=False))

    summary = build_quality_summary(market_data)
    print("\nMarket-data quality-control summary:")
    print(summary.to_string(index=False))

    save_outputs(market_data, combined_data, summary)
    if warnings:
        print("\nCompleted with warnings:")
        for warning in warnings:
            print(f"- {warning}")
    print("Market-data pipeline completed successfully.")


if __name__ == "__main__":
    main()
