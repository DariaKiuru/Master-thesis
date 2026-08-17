"""Phase 6: align post-level Reddit sentiment to each market's trading days.

Why this phase exists
    Reddit discussion occurs every calendar day, whereas the regression uses
    each exchange's actual sequence of market observations. Timing must be
    resolved before lagged predictors are created.

Main inputs
    The frozen 1,503-row post-level FinBERT file and frozen Phase 5 market/GARCH
    panel. The calendar-day Reddit file is loaded only to audit descriptives.

Main outputs
    ``data/processed/market_aligned_lagged.csv``, post-to-market mapping and
    validation diagnostics, alignment/sample tables, and an attention figure.

Methodological rules and boundaries
    Mapping starts from posts, not already-averaged calendar-day sentiment.
    For each market separately, same-day trading posts stay on that date and
    weekend/holiday posts move forward to the next actual trading date, never
    backward. Mapped posts are then equally averaged; no-post trading dates have
    attention = 0 and missing sentiment. All lags are created after the merge,
    so lag 1 means the previous actual trading observation. ``return_lag1`` uses
    decimal log return. This script defines eligible rows but estimates no model.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from config import (  # noqa: E402
    ALIGNED_REDDIT_DESCRIPTIVES_TABLE,
    ALIGNMENT_SAMPLE_SIZES_TABLE,
    ALIGNMENT_TIMING_REVIEW_SAMPLE_FILE,
    ALIGNMENT_WEIGHTING_VALIDATION_FILE,
    DAILY_REDDIT_FILE,
    DIAGNOSTICS_DIR,
    END_DATE,
    FINBERT_REDDIT_FILE,
    MARKET_ALIGNED_ATTENTION_FIGURE,
    MARKET_ALIGNED_LAGGED_FILE,
    MARKET_PRICES_FILE,
    MARKET_RETURNS_GARCH_FILE,
    MARKET_TICKERS,
    REDDIT_TRADING_DAY_MAPPING_FILE,
    REGRESSION_SAMPLE_SUPPORT_TABLE,
    SENTIMENT_MISSINGNESS_TRANSITIONS_FILE,
    START_DATE,
    TRADING_DAY_ALIGNMENT_VALIDATION_FILE,
    TRADING_DAY_COVERAGE_TABLE,
    TRADING_DAY_MAPPING_RECONCILIATION_TABLE,
    YEARLY_REGRESSION_SUPPORT_TABLE,
)


EXPECTED_FINBERT_SHA256 = (
    "2E4A693558197B8007F81C5D348362140524E3C31A8D31293F86D691DDB9C7FF"
)
EXPECTED_DAILY_SHA256 = (
    "BED7F6093F7F0CBF4912B3117434C6C3180ED88851D45B9906919A66D09BE2F1"
)
EXPECTED_MARKET_SHA256 = (
    "FFCF4D30CE1A064B04E726B67273204CF53CDBF1EE5696F1F727251966C425DE"
)
EXPECTED_PRICES_SHA256 = (
    "37342707BBB8E19FBDEF363B16C0372ED4420B22A2D7F56829BE166B162443AC"
)
EXPECTED_POSTS = 1_503
EXPECTED_CALENDAR_DAYS = 1_095
EXPECTED_POPULATED_CALENDAR_DAYS = 504
EXPECTED_ZERO_POST_DAYS = 591
EXPECTED_MARKETS = ["EURO_STOXX_50", "DAX", "CAC_40", "FTSE_100", "WIG20"]
EXPECTED_MARKET_ROWS = {
    "EURO_STOXX_50": 759,
    "DAX": 767,
    "CAC_40": 770,
    "FTSE_100": 754,
    "WIG20": 752,
}
MARKET_COLUMNS = [
    "date",
    "index_name",
    "ticker",
    "data_source",
    "close_level",
    "log_return",
    "return_pct",
    "garch_volatility",
]
PANEL_COLUMNS = [
    *MARKET_COLUMNS,
    "sentiment",
    "attention",
    "sentiment_lag1",
    "attention_lag1",
    "volatility_lag1",
    "return_lag1",
    "regression_eligible",
]
MAPPING_COLUMNS = [
    "reddit_id",
    "original_post_date",
    "index_name",
    "mapped_trading_date",
    "mapping_status",
    "mapping_reason",
    "days_forward",
    "sentiment_score",
]
APPROVED_REGRESSION_VARIABLES = [
    "garch_volatility",
    "sentiment_lag1",
    "attention_lag1",
    "volatility_lag1",
    "return_lag1",
]
FIGURE_DPI = 300
NUMERIC_TOLERANCE = 1e-12


# ---------------------------------------------------------------------------
# Load frozen post-level and market inputs; audit the descriptive calendar
# ---------------------------------------------------------------------------

def file_sha256(path: Path) -> str:
    """Return an uppercase SHA-256 digest for one file."""

    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def finite(series: pd.Series) -> pd.Series:
    """Return a Boolean mask for finite numeric observations."""

    numeric = pd.to_numeric(series, errors="coerce")
    return pd.Series(np.isfinite(numeric.to_numpy(dtype=float)), index=series.index)


def numeric_equal(left: pd.Series, right: pd.Series) -> bool:
    """Compare numeric series exactly apart from CSV-safe floating precision."""

    return bool(
        np.allclose(
            pd.to_numeric(left, errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(right, errors="coerce").to_numpy(dtype=float),
            rtol=0,
            atol=NUMERIC_TOLERANCE,
            equal_nan=True,
        )
    )


def validate_configuration() -> None:
    """Require the frozen sample, markets, and Phase 6 output locations."""

    if START_DATE != "2021-01-01" or END_DATE != "2023-12-31":
        raise ValueError("Phase 6 must cover 2021-01-01 through 2023-12-31.")
    if list(MARKET_TICKERS) != EXPECTED_MARKETS:
        raise ValueError("The approved market order or membership changed.")
    configured_paths = [
        MARKET_ALIGNED_LAGGED_FILE,
        REDDIT_TRADING_DAY_MAPPING_FILE,
        TRADING_DAY_MAPPING_RECONCILIATION_TABLE,
        TRADING_DAY_COVERAGE_TABLE,
        ALIGNED_REDDIT_DESCRIPTIVES_TABLE,
        ALIGNMENT_SAMPLE_SIZES_TABLE,
        REGRESSION_SAMPLE_SUPPORT_TABLE,
        YEARLY_REGRESSION_SUPPORT_TABLE,
        TRADING_DAY_ALIGNMENT_VALIDATION_FILE,
        ALIGNMENT_TIMING_REVIEW_SAMPLE_FILE,
        ALIGNMENT_WEIGHTING_VALIDATION_FILE,
        SENTIMENT_MISSINGNESS_TRANSITIONS_FILE,
        MARKET_ALIGNED_ATTENTION_FIGURE,
    ]
    if len(configured_paths) != len(set(configured_paths)):
        raise ValueError("Phase 6 output paths must be unique.")


def load_and_validate_inputs() -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, str]
]:
    """Load immutable post, market, and descriptive calendar inputs."""

    paths = {
        "finbert": FINBERT_REDDIT_FILE,
        "daily": DAILY_REDDIT_FILE,
        "market": MARKET_RETURNS_GARCH_FILE,
        "prices": MARKET_PRICES_FILE,
    }
    expected_hashes = {
        "finbert": EXPECTED_FINBERT_SHA256,
        "daily": EXPECTED_DAILY_SHA256,
        "market": EXPECTED_MARKET_SHA256,
        "prices": EXPECTED_PRICES_SHA256,
    }
    hashes: dict[str, str] = {}
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Required Phase 6 input is missing: {path}")
        hashes[name] = file_sha256(path)
        if hashes[name] != expected_hashes[name]:
            raise ValueError(
                f"Frozen {name} SHA-256 changed. Expected "
                f"{expected_hashes[name]}, found {hashes[name]}."
            )

    posts = pd.read_csv(
        FINBERT_REDDIT_FILE,
        dtype={"id": "string", "subreddit": "string"},
    )
    required_post_columns = {"id", "date_utc", "sentiment_score"}
    if missing := required_post_columns - set(posts.columns):
        raise ValueError(f"FinBERT input lacks columns: {sorted(missing)}")
    if len(posts) != EXPECTED_POSTS or posts["id"].nunique() != EXPECTED_POSTS:
        raise ValueError("Phase 6 requires 1,503 globally unique Reddit IDs.")
    if posts["id"].isna().any() or posts["id"].astype(str).str.strip().eq("").any():
        raise ValueError("A Reddit ID is missing or blank.")
    posts = posts.copy()
    posts["original_post_date"] = pd.to_datetime(
        posts["date_utc"], errors="coerce"
    ).dt.normalize()
    if posts["original_post_date"].isna().any() or not posts[
        "original_post_date"
    ].between(START_DATE, END_DATE, inclusive="both").all():
        raise ValueError("A Reddit post has an invalid or out-of-period date.")
    posts["sentiment_score"] = pd.to_numeric(
        posts["sentiment_score"], errors="coerce"
    )
    if not finite(posts["sentiment_score"]).all() or not posts[
        "sentiment_score"
    ].between(-1, 1, inclusive="both").all():
        raise ValueError("A post-level sentiment score is invalid.")

    market = pd.read_csv(MARKET_RETURNS_GARCH_FILE, parse_dates=["date"])
    if list(market.columns) != MARKET_COLUMNS:
        raise ValueError("Frozen Phase 5 market schema changed.")
    if market.duplicated(["index_name", "date"]).any():
        raise ValueError("Frozen market input contains duplicate market/date rows.")
    if set(market["index_name"]) != set(EXPECTED_MARKETS):
        raise ValueError("Frozen market input does not contain the five markets.")
    if not market["date"].between(START_DATE, END_DATE, inclusive="both").all():
        raise ValueError("Frozen market input contains an out-of-period date.")
    ordered_market: list[pd.DataFrame] = []
    for index_name in EXPECTED_MARKETS:
        subset = market.loc[market["index_name"].eq(index_name)].copy()
        subset = subset.sort_values("date", kind="stable").reset_index(drop=True)
        if len(subset) != EXPECTED_MARKET_ROWS[index_name]:
            raise ValueError(f"{index_name} market row count changed.")
        if not subset["date"].is_unique or not subset["date"].is_monotonic_increasing:
            raise ValueError(f"{index_name} trading dates are not unique and sorted.")
        ordered_market.append(subset)
    market = pd.concat(ordered_market, ignore_index=True)

    daily = pd.read_csv(DAILY_REDDIT_FILE, parse_dates=["date"])
    validate_descriptive_calendar(daily)
    return posts, market, daily, hashes


def validate_descriptive_calendar(daily: pd.DataFrame) -> None:
    """Audit calendar sparsity without using this file for empirical alignment.

    Calendar-day means are valid descriptives, but averaging those means after
    weekend/holiday mapping would give days rather than posts equal weight.
    """

    required = {"date", "sentiment", "post_count"}
    if missing := required - set(daily.columns):
        raise ValueError(f"Daily descriptive file lacks columns: {sorted(missing)}")
    if len(daily) != EXPECTED_CALENDAR_DAYS or daily["date"].duplicated().any():
        raise ValueError("Daily descriptive calendar no longer has 1,095 unique days.")
    zero_attention = daily["post_count"].eq(0)
    positive_attention = daily["post_count"].gt(0)
    if int(positive_attention.sum()) != EXPECTED_POPULATED_CALENDAR_DAYS:
        raise ValueError("Populated daily-sentiment day count changed.")
    if int(zero_attention.sum()) != EXPECTED_ZERO_POST_DAYS:
        raise ValueError("Zero-post calendar-day count changed.")
    if daily.loc[zero_attention, "sentiment"].notna().any():
        raise ValueError("A zero-post calendar day has an empirical sentiment value.")
    observed = pd.to_numeric(
        daily.loc[positive_attention, "sentiment"], errors="coerce"
    )
    if not np.isfinite(observed.to_numpy(dtype=float)).all() or not observed.between(
        -1, 1, inclusive="both"
    ).all():
        raise ValueError("A positive-post calendar day has invalid sentiment.")
    if int(daily["post_count"].sum()) != EXPECTED_POSTS:
        raise ValueError("Calendar-day attention no longer reconciles to 1,503 posts.")


def map_posts_to_markets(posts: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    """Map every post to the same or next actual trading date per market.

    Each exchange supplies its own observed calendar. ``searchsorted`` with
    ``side='left'`` preserves same-day matches and otherwise selects the first
    later market observation; it can never map a post backward.
    """

    mapping_groups: list[pd.DataFrame] = []
    post_dates = posts["original_post_date"].to_numpy(dtype="datetime64[ns]")
    for index_name in EXPECTED_MARKETS:
        trading_dates = (
            market.loc[market["index_name"].eq(index_name), "date"]
            .sort_values()
            .to_numpy(dtype="datetime64[ns]")
        )
        positions = np.searchsorted(trading_dates, post_dates, side="left")
        mapped_values = np.full(
            len(posts), np.datetime64("NaT", "ns"), dtype="datetime64[ns]"
        )
        eligible = positions < len(trading_dates)
        mapped_values[eligible] = trading_dates[positions[eligible]]
        mapped_dates = pd.Series(mapped_values, index=posts.index)
        days_forward = (mapped_dates - posts["original_post_date"]).dt.days.astype(
            "Int64"
        )
        terminal = mapped_dates.isna()
        same_day = days_forward.eq(0).fillna(False)
        forward = days_forward.gt(0).fillna(False)
        weekday = posts["original_post_date"].dt.dayofweek

        mapping_status = pd.Series("forward_nontrading", index=posts.index)
        mapping_status.loc[same_day] = "same_day"
        mapping_status.loc[terminal] = "terminal_unmapped"
        mapping_reason = pd.Series("weekday_nontrading", index=posts.index)
        mapping_reason.loc[same_day] = "trading_day"
        mapping_reason.loc[forward & weekday.eq(5)] = "Saturday"
        mapping_reason.loc[forward & weekday.eq(6)] = "Sunday"
        mapping_reason.loc[terminal] = "terminal_end_of_sample"

        mapping_groups.append(
            pd.DataFrame(
                {
                    "reddit_id": posts["id"].astype("string"),
                    "original_post_date": posts["original_post_date"],
                    "index_name": index_name,
                    "mapped_trading_date": mapped_dates,
                    "mapping_status": mapping_status,
                    "mapping_reason": mapping_reason,
                    "days_forward": days_forward,
                    "sentiment_score": posts["sentiment_score"].astype(float),
                }
            )
        )
    mapping = pd.concat(mapping_groups, ignore_index=True)
    return mapping.loc[:, MAPPING_COLUMNS]


def aggregate_and_merge(
    mapping: pd.DataFrame, market: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate equal-weight posts, retain market rows, then create lags.

    ``sentiment`` is the mean of mapped post scores and ``attention`` is their
    count. Lagged sentiment, attention, volatility, and decimal log return are
    formed only after market-specific sorting, so lag 1 is the previous trading
    observation and not necessarily the previous calendar date.
    """

    mapped = mapping.loc[mapping["mapping_status"].ne("terminal_unmapped")].copy()
    aggregates = (
        mapped.groupby(["index_name", "mapped_trading_date"], sort=True)
        .agg(
            sentiment=("sentiment_score", "mean"),
            attention=("reddit_id", "size"),
        )
        .reset_index()
        .rename(columns={"mapped_trading_date": "date"})
    )
    if aggregates.duplicated(["index_name", "date"]).any():
        raise ValueError("Reddit aggregates contain duplicate market/date rows.")

    panel = market.merge(
        aggregates,
        on=["index_name", "date"],
        how="left",
        validate="one_to_one",
        sort=False,
    )
    # A market date with no mapped posts has zero attention but no observed
    # sentiment. Leaving sentiment missing avoids treating silence as neutral.
    panel["attention"] = panel["attention"].fillna(0).astype("int64")
    groups: list[pd.DataFrame] = []
    for index_name in EXPECTED_MARKETS:
        subset = panel.loc[panel["index_name"].eq(index_name)].copy()
        subset = subset.sort_values("date", kind="stable").reset_index(drop=True)
        # All four predictors refer to the preceding row of this market's
        # actual trading calendar, protecting the regression from look-ahead.
        subset["sentiment_lag1"] = subset["sentiment"].shift(1)
        subset["attention_lag1"] = subset["attention"].shift(1)
        subset["volatility_lag1"] = subset["garch_volatility"].shift(1)
        subset["return_lag1"] = subset["log_return"].shift(1)
        eligible = pd.Series(True, index=subset.index)
        for variable in APPROVED_REGRESSION_VARIABLES:
            eligible &= finite(subset[variable])
        subset["regression_eligible"] = eligible.astype(bool)
        groups.append(subset)
    panel = pd.concat(groups, ignore_index=True).loc[:, PANEL_COLUMNS]
    return aggregates, panel


def series_statistics(series: pd.Series) -> dict[str, float | int]:
    """Return stable descriptive statistics for one observed series."""

    description = series.describe(percentiles=[0.25, 0.50, 0.75])
    return {
        "n": int(description["count"]),
        "mean": float(description["mean"]),
        "standard_deviation": float(description["std"]),
        "minimum": float(description["min"]),
        "percentile_25": float(description["25%"]),
        "median": float(description["50%"]),
        "percentile_75": float(description["75%"]),
        "maximum": float(description["max"]),
    }


def build_mapping_reconciliation(mapping: pd.DataFrame) -> pd.DataFrame:
    """Reconcile post-level mapping statuses and forward distances by market."""

    rows: list[dict[str, Any]] = []
    for index_name in EXPECTED_MARKETS:
        subset = mapping.loc[mapping["index_name"].eq(index_name)]
        mapped = subset.loc[subset["mapping_status"].ne("terminal_unmapped")]
        rows.append(
            {
                "index_name": index_name,
                "input_posts": len(subset),
                "same_day_mapped": int(subset["mapping_status"].eq("same_day").sum()),
                "forward_mapped": int(
                    subset["mapping_status"].eq("forward_nontrading").sum()
                ),
                "saturday_forwarded": int(subset["mapping_reason"].eq("Saturday").sum()),
                "sunday_forwarded": int(subset["mapping_reason"].eq("Sunday").sum()),
                "weekday_nontrading_forwarded": int(
                    subset["mapping_reason"].eq("weekday_nontrading").sum()
                ),
                "terminal_unmapped": int(
                    subset["mapping_status"].eq("terminal_unmapped").sum()
                ),
                "total_mapped": len(mapped),
                "unique_mapped_reddit_ids": int(mapped["reddit_id"].nunique()),
                "duplicate_within_market_mappings": int(
                    subset.duplicated(["reddit_id"]).sum()
                ),
                "minimum_days_forward": int(mapped["days_forward"].min()),
                "mean_days_forward": float(mapped["days_forward"].mean()),
                "maximum_days_forward": int(mapped["days_forward"].max()),
            }
        )
    return pd.DataFrame(rows)


def build_trading_day_coverage(panel: pd.DataFrame) -> pd.DataFrame:
    """Summarize aligned Reddit coverage on each full market calendar."""

    rows: list[dict[str, Any]] = []
    for index_name in EXPECTED_MARKETS:
        subset = panel.loc[panel["index_name"].eq(index_name)]
        maximum_attention = int(subset["attention"].max())
        maximum_date = subset.loc[
            subset["attention"].eq(maximum_attention), "date"
        ].min()
        zero_attention = subset["attention"].eq(0)
        rows.append(
            {
                "index_name": index_name,
                "total_trading_dates": len(subset),
                "finite_garch_volatility_dates": int(
                    finite(subset["garch_volatility"]).sum()
                ),
                "populated_aligned_sentiment_dates": int(
                    subset["sentiment"].notna().sum()
                ),
                "zero_attention_trading_dates": int(zero_attention.sum()),
                "zero_attention_share": float(zero_attention.mean()),
                "total_mapped_posts": int(subset["attention"].sum()),
                "missing_contemporaneous_sentiment_dates": int(
                    subset["sentiment"].isna().sum()
                ),
                "first_trading_date": subset["date"].min(),
                "last_trading_date": subset["date"].max(),
                "maximum_attention": maximum_attention,
                "maximum_attention_date": maximum_date,
            }
        )
    return pd.DataFrame(rows)


def build_aligned_descriptives(panel: pd.DataFrame) -> pd.DataFrame:
    """Describe populated-day sentiment and full-calendar attention separately."""

    rows: list[dict[str, Any]] = []
    for index_name in EXPECTED_MARKETS:
        subset = panel.loc[panel["index_name"].eq(index_name)]
        sentiment = subset["sentiment"].dropna()
        attention = subset["attention"]
        rows.append(
            {
                "index_name": index_name,
                "variable": "sentiment",
                "denominator": "populated trading days only",
                "total_posts": np.nan,
                **series_statistics(sentiment),
                "maximum_date": subset.loc[subset["sentiment"].eq(sentiment.max()), "date"].min(),
            }
        )
        rows.append(
            {
                "index_name": index_name,
                "variable": "attention",
                "denominator": "complete market trading calendar",
                "total_posts": int(attention.sum()),
                **series_statistics(attention),
                "maximum_date": subset.loc[subset["attention"].eq(attention.max()), "date"].min(),
            }
        )
    return pd.DataFrame(rows)


def technical_support_mask(subset: pd.DataFrame) -> pd.Series:
    """Identify rows usable on market controls before requiring sentiment_lag1."""

    return (
        finite(subset["garch_volatility"])
        & finite(subset["attention_lag1"])
        & finite(subset["volatility_lag1"])
        & finite(subset["return_lag1"])
    )


def build_sample_sizes(panel: pd.DataFrame) -> pd.DataFrame:
    """Quantify structural, overlapping, and sentiment-specific attrition."""

    rows: list[dict[str, Any]] = []
    for index_name in EXPECTED_MARKETS:
        subset = panel.loc[panel["index_name"].eq(index_name)].reset_index(drop=True)
        finite_volatility = finite(subset["garch_volatility"])
        technical = technical_support_mask(subset)
        sentiment_missing = ~finite(subset["sentiment_lag1"])
        specifically_sentiment = technical & sentiment_missing
        eligible = subset["regression_eligible"]
        rows.append(
            {
                "index_name": index_name,
                "total_market_trading_rows": len(subset),
                "finite_current_garch_volatility_rows": int(finite_volatility.sum()),
                "structurally_unavailable_first_return_garch_rows": int(
                    ((~finite(subset["log_return"])) | (~finite_volatility)).sum()
                ),
                "rows_with_available_prior_trading_row": max(len(subset) - 1, 0),
                "rows_with_nonmissing_sentiment_lag1": int(
                    finite(subset["sentiment_lag1"]).sum()
                ),
                "rows_with_nonmissing_attention_lag1": int(
                    finite(subset["attention_lag1"]).sum()
                ),
                "rows_with_nonmissing_volatility_lag1": int(
                    finite(subset["volatility_lag1"]).sum()
                ),
                "rows_with_nonmissing_return_lag1": int(
                    finite(subset["return_lag1"]).sum()
                ),
                "otherwise_technically_usable_rows": int(technical.sum()),
                "finite_volatility_rows_with_missing_sentiment_lag1": int(
                    (finite_volatility & sentiment_missing).sum()
                ),
                "rows_excluded_specifically_because_sentiment_lag1_missing": int(
                    specifically_sentiment.sum()
                ),
                "share_otherwise_technically_usable_lost_missing_sentiment": float(
                    specifically_sentiment.sum() / technical.sum()
                ),
                "rows_complete_all_approved_phase7_variables": int(eligible.sum()),
                "regression_eligible_rows": int(eligible.sum()),
                "percentage_finite_volatility_rows_retained": float(
                    100 * eligible.sum() / finite_volatility.sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def build_yearly_support(panel: pd.DataFrame) -> pd.DataFrame:
    """Report later-regression eligibility separately for each sample year."""

    rows: list[dict[str, Any]] = []
    for index_name in EXPECTED_MARKETS:
        market_subset = panel.loc[panel["index_name"].eq(index_name)]
        for year in [2021, 2022, 2023]:
            subset = market_subset.loc[market_subset["date"].dt.year.eq(year)]
            finite_volatility = int(finite(subset["garch_volatility"]).sum())
            sentiment_rows = int(finite(subset["sentiment_lag1"]).sum())
            eligible = int(subset["regression_eligible"].sum())
            rows.append(
                {
                    "index_name": index_name,
                    "year": year,
                    "finite_volatility_rows": finite_volatility,
                    "rows_with_sentiment_lag1": sentiment_rows,
                    "regression_eligible_rows": eligible,
                    "retention_rate": float(eligible / finite_volatility),
                }
            )
    return pd.DataFrame(rows)


def build_regression_support(panel: pd.DataFrame) -> pd.DataFrame:
    """Describe sentiment and attention support without estimating regressions."""

    rows: list[dict[str, Any]] = []
    for index_name in EXPECTED_MARKETS:
        eligible = panel.loc[
            panel["index_name"].eq(index_name) & panel["regression_eligible"]
        ]
        sentiment = eligible["sentiment_lag1"]
        attention = eligible["attention_lag1"]
        attention_zero = int(attention.eq(0).sum())
        attention_one = int(attention.eq(1).sum())
        attention_more = int(attention.gt(1).sum())
        rows.append(
            {
                "index_name": index_name,
                "regression_eligible_n": len(eligible),
                "sentiment_lag1_mean": float(sentiment.mean()),
                "sentiment_lag1_standard_deviation": float(sentiment.std()),
                "sentiment_lag1_minimum": float(sentiment.min()),
                "sentiment_lag1_median": float(sentiment.median()),
                "sentiment_lag1_maximum": float(sentiment.max()),
                "sentiment_lag1_distinct_values": int(sentiment.nunique()),
                "attention_lag1_mean": float(attention.mean()),
                "attention_lag1_standard_deviation": float(attention.std()),
                "attention_lag1_minimum": float(attention.min()),
                "attention_lag1_percentile_25": float(attention.quantile(0.25)),
                "attention_lag1_median": float(attention.median()),
                "attention_lag1_percentile_75": float(attention.quantile(0.75)),
                "attention_lag1_maximum": float(attention.max()),
                "attention_lag1_distinct_values": int(attention.nunique()),
                "attention_lag1_equals_0_rows": attention_zero,
                "attention_lag1_equals_1_rows": attention_one,
                "attention_lag1_greater_than_1_rows": attention_more,
                "attention_lag1_greater_than_0_share": float(attention.gt(0).mean()),
            }
        )
    return pd.DataFrame(rows)


def longest_true_run(mask: pd.Series, dates: pd.Series) -> tuple[int, Any, Any]:
    """Return length and date bounds of the earliest longest consecutive run."""

    values = mask.fillna(False).to_numpy(dtype=bool)
    best_start = -1
    best_length = 0
    current_start = -1
    for position, value in enumerate(values):
        if value and current_start < 0:
            current_start = position
        if current_start >= 0 and (not value or position == len(values) - 1):
            end_position = position if value and position == len(values) - 1 else position - 1
            run_length = end_position - current_start + 1
            if run_length > best_length:
                best_start = current_start
                best_length = run_length
            current_start = -1
    if best_length == 0:
        return 0, pd.NaT, pd.NaT
    return (
        best_length,
        dates.iloc[best_start],
        dates.iloc[best_start + best_length - 1],
    )


def build_missingness_transitions(panel: pd.DataFrame) -> pd.DataFrame:
    """Describe how zero-attention sentiment missingness propagates through lagging."""

    rows: list[dict[str, Any]] = []
    for index_name in EXPECTED_MARKETS:
        subset = panel.loc[panel["index_name"].eq(index_name)].reset_index(drop=True)
        prior_observed = subset["sentiment"].shift(1).notna()
        prior_missing = subset["sentiment"].shift(1).isna()
        has_prior = pd.Series(np.arange(len(subset)) > 0, index=subset.index)
        observed_lag = subset["sentiment_lag1"].notna()
        missing_lag = subset["sentiment_lag1"].isna()
        zero_attention = subset["attention"].eq(0)
        technical = technical_support_mask(subset)
        missing_due_prior_no_discussion = technical & missing_lag
        zero_length, zero_start, zero_end = longest_true_run(
            zero_attention, subset["date"]
        )
        missing_length, missing_start, missing_end = longest_true_run(
            missing_due_prior_no_discussion, subset["date"]
        )
        rows.append(
            {
                "index_name": index_name,
                "trading_days": len(subset),
                "zero_attention_trading_days": int(zero_attention.sum()),
                "prior_observed_to_observed_sentiment_lag1": int(
                    (has_prior & prior_observed & observed_lag).sum()
                ),
                "prior_missing_to_missing_sentiment_lag1": int(
                    (has_prior & prior_missing & missing_lag).sum()
                ),
                "subsequent_rows_ineligible_due_missing_prior_sentiment": int(
                    missing_due_prior_no_discussion.sum()
                ),
                "consecutive_zero_attention_days_occur": bool(zero_length > 1),
                "longest_consecutive_zero_attention_run": zero_length,
                "longest_zero_attention_run_start": zero_start,
                "longest_zero_attention_run_end": zero_end,
                "longest_ineligible_run_due_missing_prior_sentiment": missing_length,
                "longest_missing_prior_sentiment_run_start": missing_start,
                "longest_missing_prior_sentiment_run_end": missing_end,
            }
        )
    return pd.DataFrame(rows)


def build_timing_review_sample(
    mapping: pd.DataFrame, panel: pd.DataFrame
) -> pd.DataFrame:
    """Select deterministic cases showing mapping, aggregation, and next-row lagging."""

    aggregate_lookup = panel[
        ["index_name", "date", "sentiment", "attention"]
    ].rename(
        columns={
            "date": "mapped_trading_date",
            "sentiment": "aligned_sentiment",
            "attention": "aligned_attention",
        }
    )
    calendar_rows: list[pd.DataFrame] = []
    for index_name in EXPECTED_MARKETS:
        subset = panel.loc[panel["index_name"].eq(index_name)].sort_values("date")
        calendar_rows.append(
            pd.DataFrame(
                {
                    "index_name": index_name,
                    "mapped_trading_date": subset["date"].to_numpy(),
                    "next_trading_date": subset["date"].shift(-1).to_numpy(),
                }
            )
        )
    calendar = pd.concat(calendar_rows, ignore_index=True)
    next_rows = panel[
        ["index_name", "date", "sentiment_lag1", "attention_lag1"]
    ].rename(
        columns={
            "date": "next_trading_date",
            "sentiment_lag1": "next_row_sentiment_lag1",
            "attention_lag1": "next_row_attention_lag1",
        }
    )
    enriched = mapping.merge(
        aggregate_lookup,
        on=["index_name", "mapped_trading_date"],
        how="left",
        validate="many_to_one",
    ).merge(
        calendar,
        on=["index_name", "mapped_trading_date"],
        how="left",
        validate="many_to_one",
    ).merge(
        next_rows,
        on=["index_name", "next_trading_date"],
        how="left",
        validate="many_to_one",
    )
    enriched["next_row_receives_aligned_sentiment"] = np.isclose(
        enriched["aligned_sentiment"],
        enriched["next_row_sentiment_lag1"],
        rtol=0,
        atol=NUMERIC_TOLERANCE,
        equal_nan=False,
    )
    enriched["next_row_receives_aligned_attention"] = np.isclose(
        enriched["aligned_attention"],
        enriched["next_row_attention_lag1"],
        rtol=0,
        atol=0,
        equal_nan=False,
    )
    enriched = enriched.sort_values(
        ["original_post_date", "reddit_id", "index_name"], kind="stable"
    )

    selections: list[pd.DataFrame] = []

    def take_first(label: str, mask: pd.Series, rule: str) -> None:
        candidate = enriched.loc[mask].head(1).copy()
        if not candidate.empty:
            candidate["selection_category"] = label
            candidate["selection_rule"] = rule
            selections.append(candidate)

    take_first(
        "same_day",
        enriched["mapping_status"].eq("same_day"),
        "Earliest post-ID row with a same-day mapping.",
    )
    take_first(
        "Saturday",
        enriched["mapping_reason"].eq("Saturday"),
        "Earliest post-ID row forwarded from Saturday.",
    )
    take_first(
        "Sunday",
        enriched["mapping_reason"].eq("Sunday"),
        "Earliest post-ID row forwarded from Sunday.",
    )
    take_first(
        "weekday_nontrading",
        enriched["mapping_reason"].eq("weekday_nontrading"),
        "Earliest post-ID row forwarded from a weekday absent from the market calendar.",
    )
    take_first(
        "February_March_2022",
        enriched["original_post_date"].between("2022-02-01", "2022-03-31"),
        "Earliest post-ID row in February or March 2022.",
    )

    mapped_only = enriched.dropna(subset=["mapped_trading_date"])
    differing_ids = (
        mapped_only.groupby("reddit_id")["mapped_trading_date"]
        .nunique()
        .loc[lambda values: values.gt(1)]
        .index
    )
    if len(differing_ids):
        first_id = (
            mapped_only.loc[mapped_only["reddit_id"].isin(differing_ids)]
            .sort_values(["original_post_date", "reddit_id"], kind="stable")
            ["reddit_id"]
            .iloc[0]
        )
        different = mapped_only.loc[mapped_only["reddit_id"].eq(first_id)].copy()
        different["selection_category"] = "different_market_mapping"
        different["selection_rule"] = (
            "All market rows for the earliest Reddit ID mapping to multiple dates."
        )
        selections.append(different)

    terminal = enriched.loc[
        enriched["mapping_status"].eq("terminal_unmapped")
    ].groupby("index_name", sort=False).head(1).copy()
    if not terminal.empty:
        terminal["selection_category"] = "terminal_unmapped"
        terminal["selection_rule"] = "Earliest terminal-unmapped row per market."
        selections.append(terminal)

    if not selections:
        raise ValueError("No deterministic timing-review cases were available.")
    sample = pd.concat(selections, ignore_index=True)
    columns = [
        "selection_category",
        "selection_rule",
        "reddit_id",
        "original_post_date",
        "index_name",
        "mapping_status",
        "mapping_reason",
        "days_forward",
        "sentiment_score",
        "mapped_trading_date",
        "aligned_sentiment",
        "aligned_attention",
        "next_trading_date",
        "next_row_sentiment_lag1",
        "next_row_attention_lag1",
        "next_row_receives_aligned_sentiment",
        "next_row_receives_aligned_attention",
    ]
    return sample.loc[:, columns].sort_values(
        ["selection_category", "original_post_date", "index_name"], kind="stable"
    )


def build_weighting_validation(
    mapping: pd.DataFrame, panel: pd.DataFrame
) -> pd.DataFrame:
    """Prove equal post weighting for one deterministic multi-date aggregate."""

    mapped = mapping.dropna(subset=["mapped_trading_date"]).copy()
    selected: pd.DataFrame | None = None
    selection_rule = ""
    for index_name in EXPECTED_MARKETS:
        market_rows = mapped.loc[mapped["index_name"].eq(index_name)]
        for mapped_date, group in market_rows.groupby("mapped_trading_date", sort=True):
            if group["original_post_date"].nunique() < 2:
                continue
            day_groups = group.groupby("original_post_date")["sentiment_score"]
            day_counts = day_groups.size()
            correct_mean = float(group["sentiment_score"].mean())
            unweighted_day_mean = float(day_groups.mean().mean())
            if day_counts.nunique() > 1 and not np.isclose(
                correct_mean,
                unweighted_day_mean,
                rtol=0,
                atol=NUMERIC_TOLERANCE,
            ):
                selected = group.copy()
                selection_rule = (
                    "Earliest market/date with multiple source dates, unequal source-date "
                    "post counts, and a nonzero difference between post weighting and an "
                    "incorrect equal-day-mean calculation."
                )
                break
        if selected is not None:
            break
    if selected is None:
        candidates = mapped.groupby(["index_name", "mapped_trading_date"], sort=True)
        for _, group in candidates:
            if group["original_post_date"].nunique() >= 2:
                selected = group.copy()
                selection_rule = "Earliest market/date with multiple original post dates."
                break
    if selected is None:
        raise ValueError("No multi-calendar-date trading-day aggregate exists.")

    calendar_day = (
        selected.groupby("original_post_date")["sentiment_score"]
        .agg(calendar_day_post_count="size", calendar_day_sentiment="mean")
        .reset_index()
    )
    selected = selected.merge(calendar_day, on="original_post_date", validate="many_to_one")
    correct_mean = float(selected["sentiment_score"].mean())
    naive_day_mean = float(calendar_day["calendar_day_sentiment"].mean())
    panel_value = float(
        panel.loc[
            panel["index_name"].eq(selected["index_name"].iloc[0])
            & panel["date"].eq(selected["mapped_trading_date"].iloc[0]),
            "sentiment",
        ].iloc[0]
    )
    selected["post_weight"] = 1.0 / len(selected)
    selected["weighted_post_contribution"] = (
        selected["sentiment_score"] * selected["post_weight"]
    )
    selected["equal_post_weight_recomputed_sentiment"] = correct_mean
    selected["saved_aligned_sentiment"] = panel_value
    selected["incorrect_unweighted_calendar_day_mean"] = naive_day_mean
    selected["post_mean_minus_unweighted_day_mean"] = correct_mean - naive_day_mean
    selected["selection_rule"] = selection_rule
    selected["validation_formula"] = (
        "sum(individual sentiment_score) / number of mapped posts"
    )
    columns = [
        "selection_rule",
        "validation_formula",
        "index_name",
        "mapped_trading_date",
        "reddit_id",
        "original_post_date",
        "sentiment_score",
        "post_weight",
        "weighted_post_contribution",
        "calendar_day_post_count",
        "calendar_day_sentiment",
        "equal_post_weight_recomputed_sentiment",
        "saved_aligned_sentiment",
        "incorrect_unweighted_calendar_day_mean",
        "post_mean_minus_unweighted_day_mean",
    ]
    return selected.loc[:, columns].sort_values(
        ["original_post_date", "reddit_id"], kind="stable"
    )


def build_validation_diagnostic(
    posts: pd.DataFrame,
    market: pd.DataFrame,
    daily: pd.DataFrame,
    mapping: pd.DataFrame,
    aggregates: pd.DataFrame,
    panel: pd.DataFrame,
    hashes: dict[str, str],
    weighting: pd.DataFrame,
) -> pd.DataFrame:
    """Create measured PASS/FAIL checks for all frozen Phase 6 rules.

    The checks reconcile hashes and counts, prove that mapping never moves
    backward, preserve Phase 5 market variables, and reproduce every lag from
    the prior trading row. They are empirical reproducibility safeguards.
    """

    rows: list[dict[str, Any]] = []

    def add(
        check: str,
        passed: bool,
        measured: Any,
        expected: Any,
        index_name: str = "ALL",
        details: str = "",
    ) -> None:
        rows.append(
            {
                "index_name": index_name,
                "check": check,
                "status": "PASS" if bool(passed) else "FAIL",
                "measured_value": measured,
                "expected_value": expected,
                "details": details,
            }
        )

    add("frozen_finbert_hash", hashes["finbert"] == EXPECTED_FINBERT_SHA256, hashes["finbert"], EXPECTED_FINBERT_SHA256)
    add("frozen_phase5_hash", hashes["market"] == EXPECTED_MARKET_SHA256, hashes["market"], EXPECTED_MARKET_SHA256)
    add("frozen_market_prices_hash", hashes["prices"] == EXPECTED_PRICES_SHA256, hashes["prices"], EXPECTED_PRICES_SHA256)
    add("unique_input_reddit_ids", posts["id"].nunique() == EXPECTED_POSTS, posts["id"].nunique(), EXPECTED_POSTS)
    add("calendar_days_descriptive_only", len(daily) == EXPECTED_CALENDAR_DAYS, len(daily), EXPECTED_CALENDAR_DAYS, details="daily_reddit_sentiment.csv was audited but not used in mapping or aggregation.")
    add("calendar_zero_posts_imply_missing_sentiment", daily.loc[daily["post_count"].eq(0), "sentiment"].isna().all(), int(daily.loc[daily["post_count"].eq(0), "sentiment"].notna().sum()), 0)
    add("canonical_unique_market_date_rows", not panel.duplicated(["index_name", "date"]).any(), int(panel.duplicated(["index_name", "date"]).sum()), 0)
    add("canonical_total_market_rows_preserved", len(panel) == len(market), len(panel), len(market))

    for index_name in EXPECTED_MARKETS:
        mapped_subset = mapping.loc[mapping["index_name"].eq(index_name)]
        mapped = mapped_subset.loc[mapped_subset["mapping_status"].ne("terminal_unmapped")]
        market_subset = market.loc[market["index_name"].eq(index_name)].reset_index(drop=True)
        panel_subset = panel.loc[panel["index_name"].eq(index_name)].reset_index(drop=True)
        terminal_count = int(mapped_subset["mapping_status"].eq("terminal_unmapped").sum())
        no_backward = bool(
            mapped["mapped_trading_date"].ge(mapped["original_post_date"]).all()
        )
        actual_dates = set(market_subset["date"])
        mapped_dates_actual = bool(mapped["mapped_trading_date"].isin(actual_dates).all())
        duplicate_count = int(mapped_subset.duplicated(["reddit_id"]).sum())
        attention_total = int(panel_subset["attention"].sum())
        aggregate_total = int(
            aggregates.loc[aggregates["index_name"].eq(index_name), "attention"].sum()
        )
        zero_mask = panel_subset["attention"].eq(0)
        positive_mask = panel_subset["attention"].gt(0)
        valid_positive_sentiment = (
            finite(panel_subset.loc[positive_mask, "sentiment"]).all()
            and panel_subset.loc[positive_mask, "sentiment"].between(-1, 1).all()
        )
        add("mapping_never_backward", no_backward, int((mapped["mapped_trading_date"] < mapped["original_post_date"]).sum()), 0, index_name)
        add("mapped_dates_are_actual_market_dates", mapped_dates_actual, int((~mapped["mapped_trading_date"].isin(actual_dates)).sum()), 0, index_name)
        add("max_one_mapping_per_post_market", duplicate_count == 0, duplicate_count, 0, index_name)
        add("mapping_reconciliation", len(mapped_subset) == len(mapped) + terminal_count, f"{len(mapped_subset)}={len(mapped)}+{terminal_count}", f"{EXPECTED_POSTS}=mapped+terminal", index_name)
        add("aggregate_attention_reconciliation", attention_total == len(mapped) == aggregate_total, f"panel={attention_total}; aggregate={aggregate_total}; mapped={len(mapped)}", len(mapped), index_name)
        add("market_row_count_preserved", len(panel_subset) == len(market_subset), len(panel_subset), len(market_subset), index_name)
        add("zero_attention_implies_missing_sentiment", panel_subset.loc[zero_mask, "sentiment"].isna().all(), int(panel_subset.loc[zero_mask, "sentiment"].notna().sum()), 0, index_name)
        add("positive_attention_implies_finite_sentiment", valid_positive_sentiment, int((~finite(panel_subset.loc[positive_mask, "sentiment"])).sum()), 0, index_name)
        add("observed_sentiment_within_minus1_plus1", panel_subset.loc[positive_mask, "sentiment"].between(-1, 1).all(), f"min={panel_subset['sentiment'].min()}; max={panel_subset['sentiment'].max()}", "[-1, 1]", index_name)
        for variable in ["close_level", "log_return", "return_pct", "garch_volatility"]:
            add(f"frozen_{variable}_preserved", numeric_equal(panel_subset[variable], market_subset[variable]), "exact within tolerance", "match Phase 5", index_name)
        lag_pairs = {
            "sentiment_lag1": "sentiment",
            "attention_lag1": "attention",
            "volatility_lag1": "garch_volatility",
            "return_lag1": "log_return",
        }
        for lag, source in lag_pairs.items():
            add(f"{lag}_reproduces_shift1", numeric_equal(panel_subset[lag], panel_subset[source].shift(1)), "within-market shift(1)", "exact shift(1)", index_name)
        no_lookahead = bool(
            panel_subset["date"].iloc[1:].reset_index(drop=True).gt(
                panel_subset["date"].shift(1).iloc[1:].reset_index(drop=True)
            ).all()
        )
        add("no_lookahead_lag_source_date", no_lookahead, int((panel_subset["date"].diff().iloc[1:] <= pd.Timedelta(0)).sum()), 0, index_name)
        eligible = panel_subset["regression_eligible"]
        eligible_finite = all(
            finite(panel_subset.loc[eligible, variable]).all()
            for variable in APPROVED_REGRESSION_VARIABLES
        )
        recomputed = pd.Series(True, index=panel_subset.index)
        for variable in APPROVED_REGRESSION_VARIABLES:
            recomputed &= finite(panel_subset[variable])
        add("eligible_rows_all_finite", eligible_finite, int(eligible.sum()), "all approved variables finite", index_name)
        add("regression_eligibility_reproduces", eligible.equals(recomputed), int((eligible != recomputed).sum()), 0, index_name)

    weighting_pass = bool(
        np.isclose(
            weighting["equal_post_weight_recomputed_sentiment"].iloc[0],
            weighting["saved_aligned_sentiment"].iloc[0],
            rtol=0,
            atol=NUMERIC_TOLERANCE,
        )
        and np.isclose(weighting["post_weight"].sum(), 1.0)
    )
    add("equal_post_weighting_example", weighting_pass, weighting["equal_post_weight_recomputed_sentiment"].iloc[0], weighting["saved_aligned_sentiment"].iloc[0])
    validation = pd.DataFrame(rows)
    failures = validation.loc[validation["status"].eq("FAIL")]
    if not failures.empty:
        failed_names = failures[["index_name", "check"]].astype(str).agg(": ".join, axis=1)
        raise RuntimeError("Phase 6 validation failed: " + "; ".join(failed_names))
    return validation


def configure_plots() -> None:
    """Apply the restrained thesis plotting style."""

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "axes.edgecolor": "#4B5563",
            "axes.linewidth": 0.8,
            "grid.color": "#D1D5DB",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.65,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def plot_aligned_attention(panel: pd.DataFrame) -> None:
    """Plot unsmoothed aligned attention on actual dates for all markets."""

    colors = {
        "EURO_STOXX_50": "#2563EB",
        "DAX": "#7C3AED",
        "CAC_40": "#0F766E",
        "FTSE_100": "#C2410C",
        "WIG20": "#B91C1C",
    }
    configure_plots()
    fig, axes = plt.subplots(5, 1, figsize=(11, 11), sharex=True, sharey=True)
    for ax, index_name in zip(axes, EXPECTED_MARKETS, strict=True):
        subset = panel.loc[panel["index_name"].eq(index_name)]
        ax.plot(
            subset["date"],
            subset["attention"],
            color=colors[index_name],
            linewidth=0.75,
        )
        ax.set_ylim(bottom=0)
        ax.set_title(index_name.replace("_", " "), loc="left", fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(pd.Timestamp(START_DATE), pd.Timestamp(END_DATE))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
        ax.tick_params(axis="x", which="minor", labelbottom=False)
    axes[-1].set_xlabel("Trading date")
    fig.supylabel("Mapped Reddit posts (count)", x=0.015)
    fig.suptitle("Market-specific aligned Reddit attention", fontsize=14)
    fig.text(
        0.5,
        0.012,
        "Actual market trading dates; zero-attention days remain zero; no smoothing, interpolation, or normalization.",
        ha="center",
        fontsize=8.5,
        color="#374151",
    )
    fig.tight_layout(rect=[0.035, 0.03, 1, 0.97])
    MARKET_ALIGNED_ATTENTION_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        MARKET_ALIGNED_ATTENTION_FIGURE,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )
    plt.close(fig)


def write_csv(data: pd.DataFrame, path: Path) -> None:
    """Write one validated Phase 6 CSV with explicit missing values."""

    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False, date_format="%Y-%m-%d", na_rep="")


def main() -> None:
    """Run Phase 6 only; do not estimate regressions or modify frozen inputs."""

    validate_configuration()
    posts, market, daily, hashes = load_and_validate_inputs()
    mapping = map_posts_to_markets(posts, market)
    aggregates, panel = aggregate_and_merge(mapping, market)

    mapping_reconciliation = build_mapping_reconciliation(mapping)
    coverage = build_trading_day_coverage(panel)
    descriptives = build_aligned_descriptives(panel)
    sample_sizes = build_sample_sizes(panel)
    yearly_support = build_yearly_support(panel)
    regression_support = build_regression_support(panel)
    missingness = build_missingness_transitions(panel)
    timing_sample = build_timing_review_sample(mapping, panel)
    weighting = build_weighting_validation(mapping, panel)
    validation = build_validation_diagnostic(
        posts,
        market,
        daily,
        mapping,
        aggregates,
        panel,
        hashes,
        weighting,
    )

    outputs = {
        MARKET_ALIGNED_LAGGED_FILE: panel,
        REDDIT_TRADING_DAY_MAPPING_FILE: mapping,
        TRADING_DAY_MAPPING_RECONCILIATION_TABLE: mapping_reconciliation,
        TRADING_DAY_COVERAGE_TABLE: coverage,
        ALIGNED_REDDIT_DESCRIPTIVES_TABLE: descriptives,
        ALIGNMENT_SAMPLE_SIZES_TABLE: sample_sizes,
        REGRESSION_SAMPLE_SUPPORT_TABLE: regression_support,
        YEARLY_REGRESSION_SUPPORT_TABLE: yearly_support,
        TRADING_DAY_ALIGNMENT_VALIDATION_FILE: validation,
        ALIGNMENT_TIMING_REVIEW_SAMPLE_FILE: timing_sample,
        ALIGNMENT_WEIGHTING_VALIDATION_FILE: weighting,
        SENTIMENT_MISSINGNESS_TRANSITIONS_FILE: missingness,
    }
    for path, data in outputs.items():
        write_csv(data, path)
    plot_aligned_attention(panel)

    if file_sha256(FINBERT_REDDIT_FILE) != EXPECTED_FINBERT_SHA256:
        raise RuntimeError("Frozen FinBERT input changed during Phase 6.")
    if file_sha256(MARKET_RETURNS_GARCH_FILE) != EXPECTED_MARKET_SHA256:
        raise RuntimeError("Frozen Phase 5 input changed during Phase 6.")
    required_paths = [*outputs, MARKET_ALIGNED_ATTENTION_FIGURE]
    if missing := [path for path in required_paths if not path.exists()]:
        raise RuntimeError(f"Phase 6 outputs are missing: {missing}")
    if empty := [path for path in required_paths if path.stat().st_size == 0]:
        raise RuntimeError(f"Phase 6 outputs are empty: {empty}")

    print(
        "Pre-alignment calendar audit: "
        f"days={len(daily)}, populated={daily['sentiment'].notna().sum()}, "
        f"zero-post={daily['post_count'].eq(0).sum()} "
        f"({daily['post_count'].eq(0).mean():.2%})."
    )
    print(mapping_reconciliation.to_string(index=False))
    print("\nRegression-eligible sample sizes:")
    print(
        sample_sizes[
            [
                "index_name",
                "finite_current_garch_volatility_rows",
                "rows_excluded_specifically_because_sentiment_lag1_missing",
                "regression_eligible_rows",
                "percentage_finite_volatility_rows_retained",
            ]
        ].to_string(index=False)
    )
    print(f"Saved canonical Phase 6 panel to {MARKET_ALIGNED_LAGGED_FILE}")
    print(f"Canonical Phase 6 SHA-256: {file_sha256(MARKET_ALIGNED_LAGGED_FILE)}")
    print("Phase 6 alignment and lag construction completed without regressions.")


if __name__ == "__main__":
    main()
