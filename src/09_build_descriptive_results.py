"""Build retrospective descriptive results from frozen validated datasets."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

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
    CLEANED_REDDIT_FILE,
    DAILY_REDDIT_ATTENTION_FIGURE,
    DAILY_REDDIT_DESCRIPTIVES_TABLE,
    DAILY_REDDIT_EXTREMES_FILE,
    DAILY_REDDIT_FILE,
    DAILY_REDDIT_SENTIMENT_FIGURE,
    DAILY_REDDIT_SUMMARY_FILE,
    DESCRIPTIVE_RESULTS_VALIDATION_FILE,
    FINBERT_CLASS_DISTRIBUTION_FILE,
    FINBERT_LABEL_DISTRIBUTION_FIGURE,
    FINBERT_POST_DESCRIPTIVES_TABLE,
    FINBERT_PROCESSING_DIAGNOSTICS_TABLE,
    FINBERT_REDDIT_FILE,
    FINBERT_SENTIMENT_SCORE_FIGURE,
    FINBERT_SENTIMENT_SUMMARY_FILE,
    MARKET_DATA_SUMMARY_FILE,
    MARKET_PRICE_COVERAGE_TABLE,
    MARKET_PRICE_LEVELS_QC_FIGURE,
    MARKET_PRICES_FILE,
    REDDIT_CLEANING_SUMMARY_FILE,
    REDDIT_SAMPLE_COMPOSITION_FIGURE,
    REDDIT_SAMPLE_COMPOSITION_TABLE,
    REDDIT_SAMPLE_CONSTRUCTION_TABLE,
    SENTIMENT_WEIGHTING_COMPARISON_TABLE,
    STOOQ_WIG20_ARCHIVE_URL,
    STOOQ_WIG20_SYMBOL,
    YAHOO_MARKET_TICKERS,
)


EXPECTED_INPUT_HASHES = {
    MARKET_PRICES_FILE: (
        "37342707BBB8E19FBDEF363B16C0372ED4420B22A2D7F56829BE166B162443AC"
    ),
    CLEANED_REDDIT_FILE: (
        "9FA593C18509280231547E928F88003C5918BDEDD264DFAC3E0F8D0BDA46AC8D"
    ),
    FINBERT_REDDIT_FILE: (
        "2E4A693558197B8007F81C5D348362140524E3C31A8D31293F86D691DDB9C7FF"
    ),
    DAILY_REDDIT_FILE: (
        "BED7F6093F7F0CBF4912B3117434C6C3180ED88851D45B9906919A66D09BE2F1"
    ),
}
EXPECTED_POSTS = 1_503
EXPECTED_YEAR_COUNTS = {2021: 83, 2022: 1_197, 2023: 223}
EXPECTED_SUBREDDIT_COUNTS = {"stocks": 772, "StockMarket": 433, "investing": 298}
EXPECTED_LABEL_COUNTS = {"positive": 106, "neutral": 953, "negative": 444}
EXPECTED_CALENDAR_DAYS = 1_095
EXPECTED_SENTIMENT_DAYS = 504
EXPECTED_ZERO_POST_DAYS = 591
EXPECTED_MARKETS = ["EURO_STOXX_50", "DAX", "CAC_40", "FTSE_100", "WIG20"]
EXPECTED_MARKET_TICKERS = {
    **YAHOO_MARKET_TICKERS,
    "WIG20": "WIG20",
}
SOURCE_SYMBOLS = {
    **YAHOO_MARKET_TICKERS,
    "WIG20": STOOQ_WIG20_SYMBOL,
}
SENTIMENT_LABELS = ["positive", "neutral", "negative"]
PROBABILITY_COLUMNS = [
    "post_positive_probability",
    "post_neutral_probability",
    "post_negative_probability",
]
MEAN_PROBABILITY_COLUMNS = [
    "mean_positive_probability",
    "mean_neutral_probability",
    "mean_negative_probability",
]
PROBABILITY_TOLERANCE = 1e-6
SCORE_TOLERANCE = 1e-12
FIGURE_DPI = 300
EXTREME_ROWS_PER_GROUP = 10


def file_sha256(path: Path) -> str:
    """Return an uppercase SHA-256 digest for one file."""

    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def validate_input_hashes() -> dict[str, str]:
    """Stop before reporting if any canonical empirical input has changed."""

    observed: dict[str, str] = {}
    for path, expected_hash in EXPECTED_INPUT_HASHES.items():
        if not path.exists():
            raise FileNotFoundError(f"Required validated input is missing: {path}")
        observed_hash = file_sha256(path)
        observed[path.relative_to(REPOSITORY_ROOT).as_posix()] = observed_hash
        if observed_hash != expected_hash:
            raise ValueError(
                f"Immutable input changed: {path}. Expected {expected_hash}, "
                f"found {observed_hash}."
            )
    return observed


def require_columns(data: pd.DataFrame, required: set[str], name: str) -> None:
    """Require the fields used to produce one set of descriptive results."""

    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"{name} lacks required columns: {sorted(missing)}")


def parse_boolean(series: pd.Series, name: str) -> pd.Series:
    """Parse validated CSV boolean fields without treating text as truthy."""

    normalized = series.astype(str).str.strip().str.lower()
    mapping = {"true": True, "false": False}
    if not normalized.isin(mapping).all():
        raise ValueError(f"{name} contains a value other than true or false.")
    return normalized.map(mapping).astype(bool)


def diagnostic_value(
    data: pd.DataFrame,
    *,
    section: str,
    metric: str,
    grouping: str | None = None,
    group: str | None = None,
    category: str | None = None,
) -> str:
    """Read one uniquely identified value from an existing long diagnostic."""

    selected = data.loc[
        data["section"].astype(str).eq(section)
        & data["metric"].astype(str).eq(metric)
    ]
    for column, value in {
        "grouping": grouping,
        "group": group,
        "category": category,
    }.items():
        if value is not None:
            if column not in selected.columns:
                raise ValueError(f"Diagnostic does not contain {column}.")
            selected = selected.loc[selected[column].astype(str).eq(value)]
    if len(selected) != 1:
        raise ValueError(
            f"Expected one diagnostic row for {section}/{metric}, found "
            f"{len(selected)}."
        )
    return str(selected.iloc[0]["value"])


def load_inputs() -> dict[str, pd.DataFrame]:
    """Load only validated datasets and the diagnostics needed for reporting."""

    cleaned_columns = [
        "id",
        "date_utc",
        "subreddit",
        "body_status",
        "title_only",
        "relevance_path",
        "language_status",
    ]
    finbert_columns = [
        "id",
        "date_utc",
        "subreddit",
        *PROBABILITY_COLUMNS,
        "sentiment_score",
        "sentiment_label",
        "n_chunks",
        "chunk_cap_applied",
        "tokenizer_safeguard_applied",
    ]
    return {
        "cleaned": pd.read_csv(
            CLEANED_REDDIT_FILE,
            usecols=cleaned_columns,
            dtype={"id": "string", "subreddit": "string"},
        ),
        "finbert": pd.read_csv(
            FINBERT_REDDIT_FILE,
            usecols=finbert_columns,
            dtype={"id": "string", "subreddit": "string"},
        ),
        "daily": pd.read_csv(DAILY_REDDIT_FILE, parse_dates=["date"]),
        "market": pd.read_csv(MARKET_PRICES_FILE, parse_dates=["date"]),
        "cleaning_summary": pd.read_csv(
            REDDIT_CLEANING_SUMMARY_FILE, dtype="string"
        ),
        "finbert_summary": pd.read_csv(
            FINBERT_SENTIMENT_SUMMARY_FILE, dtype="string"
        ),
        "class_distribution": pd.read_csv(FINBERT_CLASS_DISTRIBUTION_FILE),
        "daily_summary": pd.read_csv(DAILY_REDDIT_SUMMARY_FILE, dtype="string"),
        "market_summary": pd.read_csv(MARKET_DATA_SUMMARY_FILE, dtype="string"),
    }


def validate_cleaned(cleaned: pd.DataFrame) -> None:
    """Reconcile the frozen final Phase 3B sample and its composition."""

    required = {
        "id",
        "date_utc",
        "subreddit",
        "body_status",
        "title_only",
        "relevance_path",
        "language_status",
    }
    require_columns(cleaned, required, "Phase 3B data")
    if len(cleaned) != EXPECTED_POSTS:
        raise ValueError(f"Expected 1,503 Phase 3B posts, found {len(cleaned):,}.")
    if cleaned["id"].isna().any() or cleaned["id"].duplicated().any():
        raise ValueError("Phase 3B IDs are missing or duplicated.")
    dates = pd.to_datetime(cleaned["date_utc"], errors="coerce")
    if dates.isna().any():
        raise ValueError("Phase 3B contains an invalid date_utc.")
    year_counts = dates.dt.year.value_counts().sort_index().to_dict()
    if year_counts != EXPECTED_YEAR_COUNTS:
        raise ValueError(f"Phase 3B annual counts changed: {year_counts}.")
    subreddit_counts = cleaned["subreddit"].value_counts().to_dict()
    if subreddit_counts != EXPECTED_SUBREDDIT_COUNTS:
        raise ValueError(f"Phase 3B subreddit counts changed: {subreddit_counts}.")
    title_only = parse_boolean(cleaned["title_only"], "title_only")
    if not title_only.eq(cleaned["body_status"].ne("available")).all():
        raise ValueError("title_only no longer reconciles to body_status.")


def validate_finbert(finbert: pd.DataFrame, cleaned: pd.DataFrame) -> None:
    """Revalidate final post-level scores without rerunning FinBERT."""

    required = {
        "id",
        "date_utc",
        "subreddit",
        *PROBABILITY_COLUMNS,
        "sentiment_score",
        "sentiment_label",
        "n_chunks",
        "chunk_cap_applied",
        "tokenizer_safeguard_applied",
    }
    require_columns(finbert, required, "Phase 4A data")
    if len(finbert) != EXPECTED_POSTS or finbert["id"].duplicated().any():
        raise ValueError("Phase 4A rows or unique IDs no longer reconcile.")
    if not finbert["id"].reset_index(drop=True).equals(
        cleaned["id"].reset_index(drop=True)
    ):
        raise ValueError("Phase 4A IDs or ordering differ from Phase 3B.")

    numeric_columns = [*PROBABILITY_COLUMNS, "sentiment_score", "n_chunks"]
    for column in numeric_columns:
        finbert[column] = pd.to_numeric(finbert[column], errors="coerce")
    if finbert[numeric_columns].isna().any().any():
        raise ValueError("Phase 4A numeric fields contain missing values.")
    probabilities = finbert[PROBABILITY_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all() or (
        (probabilities < 0) | (probabilities > 1)
    ).any():
        raise ValueError("Phase 4A probabilities are invalid.")
    if np.abs(probabilities.sum(axis=1) - 1).max() > PROBABILITY_TOLERANCE:
        raise ValueError("Phase 4A probabilities no longer sum to one.")
    scores = finbert["sentiment_score"].to_numpy(dtype=float)
    if not np.isfinite(scores).all() or ((scores < -1) | (scores > 1)).any():
        raise ValueError("Phase 4A sentiment scores are invalid.")
    if np.abs(scores - (probabilities[:, 0] - probabilities[:, 2])).max() > SCORE_TOLERANCE:
        raise ValueError("Phase 4A sentiment identity no longer holds.")
    expected_labels = np.asarray(SENTIMENT_LABELS)[probabilities.argmax(axis=1)]
    if not np.array_equal(finbert["sentiment_label"].to_numpy(), expected_labels):
        raise ValueError("Phase 4A sentiment labels no longer equal probability argmax.")
    label_counts = finbert["sentiment_label"].value_counts().to_dict()
    if label_counts != EXPECTED_LABEL_COUNTS:
        raise ValueError(f"Phase 4A label totals changed: {label_counts}.")
    if not finbert["n_chunks"].between(1, 120).all():
        raise ValueError("Phase 4A n_chunks falls outside 1 through 120.")
    parse_boolean(finbert["chunk_cap_applied"], "chunk_cap_applied")
    parse_boolean(
        finbert["tokenizer_safeguard_applied"],
        "tokenizer_safeguard_applied",
    )


def validate_daily(daily: pd.DataFrame) -> None:
    """Reconcile the complete Phase 4B calendar and missing-day treatment."""

    required = {
        "date",
        "sentiment",
        "post_count",
        "positive_post_count",
        "neutral_post_count",
        "negative_post_count",
        *MEAN_PROBABILITY_COLUMNS,
    }
    require_columns(daily, required, "Phase 4B data")
    expected_dates = pd.date_range("2021-01-01", "2023-12-31", freq="D")
    if len(daily) != EXPECTED_CALENDAR_DAYS or not np.array_equal(
        daily["date"].to_numpy(), expected_dates.to_numpy()
    ):
        raise ValueError("Phase 4B calendar coverage changed.")
    if daily["date"].duplicated().any() or not daily["date"].is_monotonic_increasing:
        raise ValueError("Phase 4B dates are duplicated or unsorted.")
    with_posts = daily["post_count"].gt(0)
    if int(with_posts.sum()) != EXPECTED_SENTIMENT_DAYS:
        raise ValueError("Phase 4B populated-day total changed.")
    if int((~with_posts).sum()) != EXPECTED_ZERO_POST_DAYS:
        raise ValueError("Phase 4B zero-post-day total changed.")
    if int(daily["post_count"].sum()) != EXPECTED_POSTS:
        raise ValueError("Phase 4B post_count no longer sums to 1,503.")
    if daily.loc[with_posts, "sentiment"].isna().any():
        raise ValueError("A populated day is missing sentiment.")
    if daily.loc[~with_posts, "sentiment"].notna().any():
        raise ValueError("A zero-post day has a sentiment value.")
    if daily.loc[~with_posts, MEAN_PROBABILITY_COLUMNS].notna().any().any():
        raise ValueError("A zero-post day has a mean probability.")
    class_columns = [
        "positive_post_count",
        "neutral_post_count",
        "negative_post_count",
    ]
    if not daily[class_columns].sum(axis=1).eq(daily["post_count"]).all():
        raise ValueError("Phase 4B class counts do not sum to post_count.")
    class_totals = {
        label: int(daily[f"{label}_post_count"].sum())
        for label in SENTIMENT_LABELS
    }
    if class_totals != EXPECTED_LABEL_COUNTS:
        raise ValueError(f"Phase 4B class totals changed: {class_totals}.")
    probabilities = daily.loc[with_posts, MEAN_PROBABILITY_COLUMNS].to_numpy(float)
    sentiment = daily.loc[with_posts, "sentiment"].to_numpy(float)
    if np.abs(probabilities.sum(axis=1) - 1).max() > PROBABILITY_TOLERANCE:
        raise ValueError("Phase 4B mean probabilities no longer sum to one.")
    if np.abs(sentiment - (probabilities[:, 0] - probabilities[:, 2])).max() > SCORE_TOLERANCE:
        raise ValueError("Phase 4B daily sentiment identity no longer holds.")
    annual_posts = daily.groupby(daily["date"].dt.year)["post_count"].sum()
    if annual_posts.astype(int).to_dict() != EXPECTED_YEAR_COUNTS:
        raise ValueError("Phase 4B annual post counts changed.")


def validate_market(market: pd.DataFrame) -> None:
    """Reconcile the five validated market level series and their sources."""

    required = {"date", "index_name", "ticker", "data_source", "close_level"}
    require_columns(market, required, "market-price data")
    if set(market["index_name"].unique()) != set(EXPECTED_MARKETS):
        raise ValueError("Market data do not contain exactly the five approved indices.")
    if market.duplicated(["index_name", "date"]).any():
        raise ValueError("Market data contain duplicate index/date combinations.")
    close_level = pd.to_numeric(market["close_level"], errors="coerce")
    if close_level.isna().any() or not np.isfinite(close_level).all():
        raise ValueError("Market close levels contain missing or non-finite values.")
    if close_level.le(0).any():
        raise ValueError("Market close levels contain non-positive values.")
    for index_name, ticker in EXPECTED_MARKET_TICKERS.items():
        subset = market.loc[market["index_name"].eq(index_name)]
        if set(subset["ticker"].astype(str)) != {ticker}:
            raise ValueError(f"Unexpected ticker for {index_name}.")
    text = market.astype(str)
    if text.apply(
        lambda column: column.str.contains("GPW.WA|WIG20.WA", case=False, regex=True)
    ).any().any():
        raise ValueError("A prohibited WIG20 proxy appears in active market data.")
    wig20 = market.loc[market["index_name"].eq("WIG20")]
    if set(wig20["data_source"].astype(str)) != {"Stooq"}:
        raise ValueError("WIG20 provenance is no longer Stooq.")


def validate_existing_diagnostics(inputs: dict[str, pd.DataFrame]) -> None:
    """Reconcile source diagnostics to canonical datasets before reporting."""

    cleaning = inputs["cleaning_summary"]
    pipeline_expected = {
        "phase3a_raw_candidates": 3_033,
        "phase3b3_relevance_retained": 1_491,
        "final_relevance_retained": 1_503,
        "confidently_non_english_removed": 0,
        "final_finbert_ready_posts": 1_503,
    }
    for metric, expected in pipeline_expected.items():
        value = int(
            float(
                diagnostic_value(
                    cleaning,
                    section="pipeline",
                    metric=metric,
                    category="all",
                )
            )
        )
        if value != expected:
            raise ValueError(f"Cleaning diagnostic changed for {metric}: {value}.")

    finbert_summary = inputs["finbert_summary"]
    chunk_expected = {
        "total_uncapped_chunks": 40_780,
        "total_scored_chunks": 34_879,
        "posts_hitting_120_chunk_cap": 69,
        "maximum_uncapped_chunks_per_post": 281,
        "conceptual_chunks_requiring_internal_fragmentation": 3,
        "total_internal_model_fragments": 8,
        "total_final_model_inputs": 34_884,
        "retained_conceptual_chunks_checked": 34_879,
    }
    for metric, expected in chunk_expected.items():
        value = int(
            float(
                diagnostic_value(
                    finbert_summary,
                    section="overall",
                    grouping="overall",
                    group="all",
                    metric=metric,
                )
            )
        )
        if value != expected:
            raise ValueError(f"FinBERT diagnostic changed for {metric}: {value}.")

    distribution = inputs["class_distribution"]
    overall = distribution.loc[distribution["grouping"].eq("overall")]
    observed_labels = overall.set_index("sentiment_label")["count"].astype(int).to_dict()
    if observed_labels != EXPECTED_LABEL_COUNTS:
        raise ValueError("FinBERT class-distribution diagnostic no longer reconciles.")

    daily_summary = inputs["daily_summary"]
    daily_expected = {
        "calendar_days": EXPECTED_CALENDAR_DAYS,
        "days_with_sentiment": EXPECTED_SENTIMENT_DAYS,
        "zero_post_days": EXPECTED_ZERO_POST_DAYS,
        "sum_post_count": EXPECTED_POSTS,
    }
    daily_sections = {
        "calendar_days": "calendar_reconciliation",
        "days_with_sentiment": "calendar_reconciliation",
        "zero_post_days": "calendar_reconciliation",
        "sum_post_count": "post_reconciliation",
    }
    for metric, expected in daily_expected.items():
        value = int(
            float(
                diagnostic_value(
                    daily_summary,
                    section=daily_sections[metric],
                    metric=metric,
                )
            )
        )
        if value != expected:
            raise ValueError(f"Daily diagnostic changed for {metric}: {value}.")

    market_summary = inputs["market_summary"]
    if set(market_summary["index_name"].astype(str)) != set(EXPECTED_MARKETS):
        raise ValueError("Market summary does not contain the five approved indices.")
    wig20 = market_summary.loc[market_summary["index_name"].eq("WIG20")]
    if len(wig20) != 1:
        raise ValueError("Market summary has an invalid WIG20 row count.")
    wig20_row = wig20.iloc[0]
    if (
        wig20_row["data_source"] != "Stooq"
        or wig20_row["retrieval_method"] != "Internet Archive snapshot of Stooq"
        or wig20_row["source_reference"] != STOOQ_WIG20_ARCHIVE_URL
    ):
        raise ValueError("WIG20 diagnostic provenance changed.")


def series_statistics(series: pd.Series) -> dict[str, float | int]:
    """Return the requested univariate descriptive statistics."""

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


def build_sample_construction(cleaning: pd.DataFrame) -> pd.DataFrame:
    """Build the thesis-facing Phase 3A-to-FinBERT sample construction table."""

    metrics = {
        metric: int(
            float(
                diagnostic_value(
                    cleaning,
                    section="pipeline",
                    metric=metric,
                    category="all",
                )
            )
        )
        for metric in [
            "phase3a_raw_candidates",
            "phase3b3_relevance_retained",
            "final_relevance_retained",
            "confidently_non_english_removed",
            "final_finbert_ready_posts",
        ]
    }
    rows = [
        {
            "order": 1,
            "metric_type": "sample_size",
            "stage_or_metric": "Phase 3A candidate posts",
            "count": metrics["phase3a_raw_candidates"],
            "change_from_previous_stage": np.nan,
            "retention_from_previous_percent": np.nan,
            "note": "Broad validated candidate corpus",
        },
        {
            "order": 2,
            "metric_type": "sample_size",
            "stage_or_metric": "Phase 3B.3 retained",
            "count": metrics["phase3b3_relevance_retained"],
            "change_from_previous_stage": (
                metrics["phase3b3_relevance_retained"]
                - metrics["phase3a_raw_candidates"]
            ),
            "retention_from_previous_percent": (
                metrics["phase3b3_relevance_retained"]
                / metrics["phase3a_raw_candidates"]
                * 100
            ),
            "note": "Validated intermediate relevance rule",
        },
        {
            "order": 3,
            "metric_type": "sample_size",
            "stage_or_metric": "Final Phase 3B retained",
            "count": metrics["final_relevance_retained"],
            "change_from_previous_stage": (
                metrics["final_relevance_retained"]
                - metrics["phase3b3_relevance_retained"]
            ),
            "retention_from_previous_percent": (
                metrics["final_relevance_retained"]
                / metrics["phase3b3_relevance_retained"]
                * 100
            ),
            "note": "Final validated relevance refinement",
        },
        {
            "order": 4,
            "metric_type": "removed_count",
            "stage_or_metric": "Confidently non-English removed",
            "count": metrics["confidently_non_english_removed"],
            "change_from_previous_stage": np.nan,
            "retention_from_previous_percent": np.nan,
            "note": "Removal count; not a sample-size stage",
        },
        {
            "order": 5,
            "metric_type": "sample_size",
            "stage_or_metric": "Final FinBERT-ready posts",
            "count": metrics["final_finbert_ready_posts"],
            "change_from_previous_stage": (
                metrics["final_finbert_ready_posts"]
                - metrics["final_relevance_retained"]
            ),
            "retention_from_previous_percent": (
                metrics["final_finbert_ready_posts"]
                / metrics["final_relevance_retained"]
                * 100
            ),
            "note": "Immutable Phase 3B canonical sample",
        },
    ]
    result = pd.DataFrame(rows)
    result["change_from_previous_stage"] = result[
        "change_from_previous_stage"
    ].astype("Int64")
    return result


def composition_rows(
    dimension: str,
    values: pd.Series,
    category_order: list[object] | None = None,
) -> list[dict[str, object]]:
    """Count one descriptive sample-composition dimension."""

    counts = values.value_counts(dropna=False)
    order = category_order if category_order is not None else sorted(counts.index)
    rows = []
    for category in order:
        count = int(counts.get(category, 0))
        rows.append(
            {
                "dimension": dimension,
                "category": str(category),
                "count": count,
                "percentage": count / len(values) * 100,
            }
        )
    return rows


def build_sample_composition(cleaned: pd.DataFrame) -> pd.DataFrame:
    """Describe final sample composition without changing sample membership."""

    year = pd.to_datetime(cleaned["date_utc"]).dt.year
    title_only = parse_boolean(cleaned["title_only"], "title_only")
    text_type = pd.Series(
        np.where(title_only, "title only", "title + body"),
        index=cleaned.index,
    )
    rows: list[dict[str, object]] = []
    rows.extend(composition_rows("year", year, [2021, 2022, 2023]))
    rows.extend(
        composition_rows(
            "subreddit",
            cleaned["subreddit"],
            ["stocks", "StockMarket", "investing"],
        )
    )
    rows.extend(
        composition_rows("text_type", text_type, ["title only", "title + body"])
    )
    rows.extend(
        composition_rows(
            "body_status",
            cleaned["body_status"],
            ["available", "blank", "deleted", "removed"],
        )
    )
    rows.extend(
        composition_rows(
            "relevance_path",
            cleaned["relevance_path"],
            [
                "geopolitical_crisis_financial",
                "crisis_specific_financial_consequence",
                "multiple_paths",
            ],
        )
    )
    rows.extend(
        composition_rows(
            "language_status",
            cleaned["language_status"],
            ["detected_english", "uncertain", "uncertain_short"],
        )
    )
    return pd.DataFrame(rows)


def finbert_group_row(
    grouping: str,
    group: str,
    subset: pd.DataFrame,
) -> dict[str, object]:
    """Build one wide post-level sentiment descriptive row."""

    statistics = series_statistics(subset["sentiment_score"])
    counts = subset["sentiment_label"].value_counts()
    row: dict[str, object] = {
        "grouping": grouping,
        "group": group,
        "n": statistics["n"],
        "sentiment_mean": statistics["mean"],
        "sentiment_standard_deviation": statistics["standard_deviation"],
        "sentiment_minimum": statistics["minimum"],
        "sentiment_percentile_25": statistics["percentile_25"],
        "sentiment_median": statistics["median"],
        "sentiment_percentile_75": statistics["percentile_75"],
        "sentiment_maximum": statistics["maximum"],
        "mean_positive_probability": subset["post_positive_probability"].mean(),
        "mean_neutral_probability": subset["post_neutral_probability"].mean(),
        "mean_negative_probability": subset["post_negative_probability"].mean(),
    }
    for label in SENTIMENT_LABELS:
        count = int(counts.get(label, 0))
        row[f"{label}_count"] = count
        row[f"{label}_percentage"] = count / len(subset) * 100
    return row


def build_finbert_descriptives(finbert: pd.DataFrame) -> pd.DataFrame:
    """Describe final post-level FinBERT results overall and by approved groups."""

    grouped = finbert.copy()
    grouped["year"] = pd.to_datetime(grouped["date_utc"]).dt.year
    rows = [finbert_group_row("overall", "all posts", grouped)]
    for year in [2021, 2022, 2023]:
        rows.append(
            finbert_group_row(
                "year",
                str(year),
                grouped.loc[grouped["year"].eq(year)],
            )
        )
    for subreddit in ["stocks", "StockMarket", "investing"]:
        rows.append(
            finbert_group_row(
                "subreddit",
                subreddit,
                grouped.loc[grouped["subreddit"].eq(subreddit)],
            )
        )
    return pd.DataFrame(rows)


def build_processing_diagnostics(finbert_summary: pd.DataFrame) -> pd.DataFrame:
    """Extract the frozen conceptual-chunk and safeguard quality controls."""

    definitions = [
        (
            "total_pre_cap_conceptual_chunks",
            "total_uncapped_chunks",
            "conceptual chunks",
            "Conceptual chunks before applying the 120-chunk post cap",
        ),
        (
            "retained_conceptual_chunks",
            "total_scored_chunks",
            "conceptual chunks",
            "Conceptual chunks retained after the cap",
        ),
        (
            "posts_affected_by_120_chunk_cap",
            "posts_hitting_120_chunk_cap",
            "posts",
            "Posts whose conceptual chunks were capped at 120",
        ),
        (
            "maximum_pre_cap_chunks_per_post",
            "maximum_uncapped_chunks_per_post",
            "conceptual chunks",
            "Largest conceptual-chunk count before the cap",
        ),
        (
            "conceptual_chunks_requiring_tokenizer_safeguard",
            "conceptual_chunks_requiring_internal_fragmentation",
            "conceptual chunks",
            "Conceptual chunks temporarily fragmented for tokenizer safety",
        ),
        (
            "temporary_low_level_model_fragments",
            "total_internal_model_fragments",
            "model fragments",
            "Temporary fragments used for the three safeguarded chunks",
        ),
        (
            "final_model_inputs",
            "total_final_model_inputs",
            "model inputs",
            "FinBERT inputs after temporary safeguard fragmentation",
        ),
        (
            "reconstructed_conceptual_probability_vectors",
            "retained_conceptual_chunks_checked",
            "probability vectors",
            "Conceptual-chunk vectors reconstructed before post averaging",
        ),
    ]
    rows = []
    for output_metric, source_metric, unit, description in definitions:
        value = int(
            float(
                diagnostic_value(
                    finbert_summary,
                    section="overall",
                    grouping="overall",
                    group="all",
                    metric=source_metric,
                )
            )
        )
        rows.append(
            {
                "metric": output_metric,
                "value": value,
                "unit": unit,
                "description": description,
                "source_metric": source_metric,
            }
        )
    return pd.DataFrame(rows)


def daily_descriptive_row(period: str, subset: pd.DataFrame) -> dict[str, object]:
    """Build one calendar-period row of daily sentiment and attention results."""

    sentiment_statistics = series_statistics(subset["sentiment"].dropna())
    post_statistics = series_statistics(subset["post_count"])
    zero_post_days = int(subset["post_count"].eq(0).sum())
    row: dict[str, object] = {
        "period": period,
        "calendar_days": len(subset),
        "sentiment_days": int(subset["sentiment"].notna().sum()),
        "zero_post_days": zero_post_days,
        "zero_post_share_percent": zero_post_days / len(subset) * 100,
        "total_posts": int(subset["post_count"].sum()),
    }
    for metric, value in sentiment_statistics.items():
        row[f"sentiment_{metric}"] = value
    for metric, value in post_statistics.items():
        row[f"post_count_{metric}"] = value
    return row


def build_daily_descriptives(daily: pd.DataFrame) -> pd.DataFrame:
    """Describe daily sentiment on observed days and attention on all days."""

    rows = [daily_descriptive_row("all", daily)]
    for year in [2021, 2022, 2023]:
        rows.append(
            daily_descriptive_row(
                str(year),
                daily.loc[daily["date"].dt.year.eq(year)],
            )
        )
    return pd.DataFrame(rows)


def build_weighting_comparison(
    finbert: pd.DataFrame,
    daily: pd.DataFrame,
) -> pd.DataFrame:
    """Report the two requested means without changing daily sentiment."""

    observed_daily = daily["sentiment"].dropna()
    return pd.DataFrame(
        [
            {
                "descriptive_mean": "Mean post-level sentiment",
                "observations": len(finbert),
                "mean_sentiment": finbert["sentiment_score"].mean(),
                "weighting_unit": "Each post receives equal weight",
                "note": "Descriptive mean across all 1,503 post-level scores",
            },
            {
                "descriptive_mean": "Mean observed calendar-day sentiment",
                "observations": len(observed_daily),
                "mean_sentiment": observed_daily.mean(),
                "weighting_unit": "Each populated calendar day receives equal weight",
                "note": (
                    "Descriptive mean across 504 populated days; this does not "
                    "change the within-day equal-post-weight sentiment definition"
                ),
            },
        ]
    )


def ranked_daily_rows(
    daily: pd.DataFrame,
    extreme_type: str,
    sort_columns: list[str],
    ascending: list[bool],
) -> list[dict[str, object]]:
    """Return deterministic ranked daily diagnostics with underlying post counts."""

    subset = daily.dropna(subset=["sentiment"]).copy()
    ranked = subset.sort_values(
        [*sort_columns, "date"],
        ascending=[*ascending, True],
        kind="stable",
    ).head(EXTREME_ROWS_PER_GROUP)
    rows = []
    for rank, record in enumerate(ranked.itertuples(index=False), start=1):
        rows.append(
            {
                "extreme_type": extreme_type,
                "rank": rank,
                "date": record.date,
                "sentiment": record.sentiment,
                "post_count": record.post_count,
                "positive_post_count": record.positive_post_count,
                "neutral_post_count": record.neutral_post_count,
                "negative_post_count": record.negative_post_count,
            }
        )
    return rows


def build_daily_extremes(daily: pd.DataFrame) -> pd.DataFrame:
    """Report high-activity and sentiment-extreme days with sample sizes."""

    rows: list[dict[str, object]] = []
    rows.extend(
        ranked_daily_rows(
            daily,
            "highest_activity",
            ["post_count", "sentiment"],
            [False, True],
        )
    )
    rows.extend(
        ranked_daily_rows(
            daily,
            "lowest_observed_sentiment",
            ["sentiment", "post_count"],
            [True, True],
        )
    )
    rows.extend(
        ranked_daily_rows(
            daily,
            "highest_observed_sentiment",
            ["sentiment", "post_count"],
            [False, True],
        )
    )
    return pd.DataFrame(rows)


def build_market_coverage(
    market: pd.DataFrame,
    market_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Build a compact market coverage and provenance table."""

    rows = []
    for index_name in EXPECTED_MARKETS:
        subset = market.loc[market["index_name"].eq(index_name)].sort_values("date")
        source = market_summary.loc[market_summary["index_name"].eq(index_name)]
        if len(source) != 1:
            raise ValueError(f"Expected one market-summary row for {index_name}.")
        source_row = source.iloc[0]
        recomputed = {
            "number_of_observations": len(subset),
            "first_date": subset["date"].min().strftime("%Y-%m-%d"),
            "last_date": subset["date"].max().strftime("%Y-%m-%d"),
            "missing_close_level_values": int(subset["close_level"].isna().sum()),
            "duplicate_dates": int(subset["date"].duplicated().sum()),
        }
        for metric, value in recomputed.items():
            if str(source_row[metric]) != str(value):
                raise ValueError(
                    f"Market summary does not reconcile for {index_name}/{metric}."
                )
        rows.append(
            {
                "index_name": index_name,
                "source_symbol": SOURCE_SYMBOLS[index_name],
                "ticker": subset["ticker"].iloc[0],
                "data_source": subset["data_source"].iloc[0],
                "retrieval_method": source_row["retrieval_method"],
                "source_reference": source_row["source_reference"],
                **recomputed,
            }
        )
    return pd.DataFrame(rows)


def build_validation_summary(
    input_hashes: dict[str, str],
    cleaned: pd.DataFrame,
    finbert: pd.DataFrame,
    daily: pd.DataFrame,
    market: pd.DataFrame,
) -> pd.DataFrame:
    """Record the principal reporting-phase validation gates in one file."""

    rows: list[dict[str, object]] = []

    def add(section: str, metric: str, value: object) -> None:
        rows.append(
            {"section": section, "metric": metric, "value": value, "status": "PASS"}
        )

    for path, digest in input_hashes.items():
        add("immutable_input_hash", path, digest)
    add("reddit_reconciliation", "phase3b_rows", len(cleaned))
    add("reddit_reconciliation", "phase4a_rows", len(finbert))
    for label in SENTIMENT_LABELS:
        add(
            "reddit_reconciliation",
            f"{label}_label_count",
            int(finbert["sentiment_label"].eq(label).sum()),
        )
    add("calendar_reconciliation", "calendar_days", len(daily))
    add(
        "calendar_reconciliation",
        "sentiment_days",
        int(daily["sentiment"].notna().sum()),
    )
    add(
        "calendar_reconciliation",
        "zero_post_days",
        int(daily["post_count"].eq(0).sum()),
    )
    add("calendar_reconciliation", "total_posts", int(daily["post_count"].sum()))
    add("market_reconciliation", "number_of_indices", market["index_name"].nunique())
    add(
        "market_reconciliation",
        "wig20_data_source",
        market.loc[market["index_name"].eq("WIG20"), "data_source"].iloc[0],
    )
    add("market_reconciliation", "prohibited_wig20_proxy_rows", 0)
    add("scope", "market_returns_calculated", False)
    add("scope", "garch_estimated", False)
    add("scope", "trading_day_alignment_performed", False)
    return pd.DataFrame(rows)


def configure_plots() -> None:
    """Apply one restrained, thesis-readable plotting style."""

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
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


def save_figure(fig: plt.Figure, path: Path) -> None:
    """Save a deterministic high-resolution PNG and close its figure."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def annotate_bars(ax: plt.Axes, counts: list[int], total: int) -> None:
    """Annotate categorical bars with counts and shares."""

    for patch, count in zip(ax.patches, counts, strict=True):
        ax.annotate(
            f"{count:,}\n({count / total * 100:.1f}%)",
            (patch.get_x() + patch.get_width() / 2, patch.get_height()),
            ha="center",
            va="bottom",
            fontsize=8,
            xytext=(0, 4),
            textcoords="offset points",
        )


def plot_sample_composition(composition: pd.DataFrame) -> None:
    """Show final Reddit sample concentration across years and subreddits."""

    year = composition.loc[composition["dimension"].eq("year")]
    subreddit = composition.loc[composition["dimension"].eq("subreddit")]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    year_counts = year["count"].astype(int).tolist()
    subreddit_counts = subreddit["count"].astype(int).tolist()
    axes[0].bar(year["category"], year_counts, color="#2563EB", width=0.65)
    axes[0].set_title("Final Reddit sample by year")
    axes[0].set_xlabel("Calendar year")
    axes[0].set_ylabel("Relevant posts")
    annotate_bars(axes[0], year_counts, EXPECTED_POSTS)
    axes[1].bar(
        subreddit["category"],
        subreddit_counts,
        color="#0F766E",
        width=0.65,
    )
    axes[1].set_title("Final Reddit sample by subreddit")
    axes[1].set_xlabel("Subreddit")
    axes[1].set_ylabel("Relevant posts")
    annotate_bars(axes[1], subreddit_counts, EXPECTED_POSTS)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.14)
    fig.suptitle("Composition of the validated 1,503-post Reddit sample", fontsize=13)
    fig.tight_layout()
    save_figure(fig, REDDIT_SAMPLE_COMPOSITION_FIGURE)


def plot_finbert_score_distribution(finbert: pd.DataFrame) -> None:
    """Plot the continuous frozen post-level sentiment score distribution."""

    scores = finbert["sentiment_score"]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    bins = np.linspace(-1, 1, 31)
    ax.hist(scores, bins=bins, color="#2563EB", edgecolor="white", linewidth=0.5)
    ax.axvline(0, color="#374151", linewidth=1, linestyle="--", label="Zero score")
    ax.set_title("Distribution of post-level FinBERT sentiment scores")
    ax.set_xlabel("Sentiment score (positive probability − negative probability)")
    ax.set_ylabel("Number of posts")
    ax.set_xlim(-1, 1)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(
        0.99,
        0.96,
        f"N = {len(scores):,}\nMean = {scores.mean():.3f}\nMedian = {scores.median():.3f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9},
    )
    fig.tight_layout()
    save_figure(fig, FINBERT_SENTIMENT_SCORE_FIGURE)


def plot_finbert_labels(finbert: pd.DataFrame) -> None:
    """Plot frozen argmax label counts and percentages."""

    labels = ["positive", "neutral", "negative"]
    counts = [int(finbert["sentiment_label"].eq(label).sum()) for label in labels]
    colors = ["#15803D", "#6B7280", "#B91C1C"]
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.bar([label.title() for label in labels], counts, color=colors, width=0.62)
    annotate_bars(ax, counts, len(finbert))
    ax.set_title("FinBERT descriptive label distribution")
    ax.set_xlabel("Argmax post-level FinBERT label")
    ax.set_ylabel("Number of posts")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save_figure(fig, FINBERT_LABEL_DISTRIBUTION_FIGURE)


def format_time_axis(ax: plt.Axes) -> None:
    """Apply readable annual ticks to a 2021–2023 time axis."""

    ax.set_xlim(pd.Timestamp("2021-01-01"), pd.Timestamp("2023-12-31"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
    ax.tick_params(axis="x", which="minor", labelbottom=False)


def plot_daily_sentiment(daily: pd.DataFrame) -> None:
    """Plot observed daily sentiment without filling or connecting missing days."""

    observed = daily.dropna(subset=["sentiment"])
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.scatter(
        observed["date"],
        observed["sentiment"],
        s=9,
        color="#2563EB",
        alpha=0.78,
        linewidths=0,
    )
    ax.axhline(0, color="#4B5563", linewidth=0.9, linestyle="--")
    ax.set_title("Daily Reddit sentiment on observed calendar days")
    ax.set_xlabel("Calendar date")
    ax.set_ylabel("Daily mean sentiment score")
    ax.set_ylim(-1, 1)
    format_time_axis(ax)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(
        0.01,
        0.02,
        "Zero-post days remain missing; no filling, interpolation, moving average, or smoothing.",
        transform=ax.transAxes,
        fontsize=8,
        color="#374151",
    )
    fig.tight_layout()
    save_figure(fig, DAILY_REDDIT_SENTIMENT_FIGURE)


def plot_daily_attention(daily: pd.DataFrame) -> None:
    """Plot the untransformed daily post count on the complete calendar."""

    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.plot(daily["date"], daily["post_count"], color="#0F766E", linewidth=0.8)
    ax.set_title("Daily Reddit attention: relevant post count")
    ax.set_xlabel("Calendar date")
    ax.set_ylabel("Relevant posts per day")
    ax.set_ylim(bottom=0)
    format_time_axis(ax)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(
        0.01,
        0.96,
        "Complete calendar; zero-post days are shown as zero. No smoothing.",
        transform=ax.transAxes,
        va="top",
        fontsize=8,
        color="#374151",
    )
    fig.tight_layout()
    save_figure(fig, DAILY_REDDIT_ATTENTION_FIGURE)


def plot_market_levels(market: pd.DataFrame) -> None:
    """Plot original close levels in separate panels for source-quality review."""

    colors = {
        "EURO_STOXX_50": "#2563EB",
        "DAX": "#7C3AED",
        "CAC_40": "#0F766E",
        "FTSE_100": "#C2410C",
        "WIG20": "#B91C1C",
    }
    mosaic = [
        ["EURO_STOXX_50", "DAX"],
        ["CAC_40", "FTSE_100"],
        ["WIG20", "WIG20"],
    ]
    fig, axes = plt.subplot_mosaic(mosaic, figsize=(11, 10), sharex=True)
    for index_name in EXPECTED_MARKETS:
        ax = axes[index_name]
        subset = market.loc[market["index_name"].eq(index_name)].sort_values("date")
        ticker = subset["ticker"].iloc[0]
        ax.plot(
            subset["date"],
            subset["close_level"],
            color=colors[index_name],
            linewidth=1.0,
        )
        ax.set_title(f"{index_name.replace('_', ' ')} ({ticker})")
        ax.set_ylabel("Close level")
        format_time_axis(ax)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("European equity-index price histories: data quality control", fontsize=14)
    fig.text(
        0.5,
        0.015,
        "Original close levels; panels use independent vertical scales. Descriptive quality-control only.",
        ha="center",
        fontsize=8.5,
        color="#374151",
    )
    fig.tight_layout(rect=[0, 0.035, 1, 0.965])
    save_figure(fig, MARKET_PRICE_LEVELS_QC_FIGURE)


def write_csv(data: pd.DataFrame, path: Path) -> None:
    """Write one machine-readable result table under the approved output tree."""

    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False, date_format="%Y-%m-%d")


def validate_generated_outputs(paths: list[Path]) -> None:
    """Require every requested reporting artifact to exist and be non-empty."""

    missing = [path for path in paths if not path.exists()]
    empty = [path for path in paths if path.exists() and path.stat().st_size == 0]
    if missing or empty:
        raise ValueError(f"Missing outputs: {missing}; empty outputs: {empty}.")


def main() -> None:
    """Generate tables and figures without changing any empirical dataset."""

    hashes_before = validate_input_hashes()
    inputs = load_inputs()
    validate_cleaned(inputs["cleaned"])
    validate_finbert(inputs["finbert"], inputs["cleaned"])
    validate_daily(inputs["daily"])
    validate_market(inputs["market"])
    validate_existing_diagnostics(inputs)

    sample_construction = build_sample_construction(inputs["cleaning_summary"])
    sample_composition = build_sample_composition(inputs["cleaned"])
    finbert_descriptives = build_finbert_descriptives(inputs["finbert"])
    processing_diagnostics = build_processing_diagnostics(inputs["finbert_summary"])
    daily_descriptives = build_daily_descriptives(inputs["daily"])
    weighting_comparison = build_weighting_comparison(
        inputs["finbert"], inputs["daily"]
    )
    daily_extremes = build_daily_extremes(inputs["daily"])
    market_coverage = build_market_coverage(
        inputs["market"], inputs["market_summary"]
    )

    table_outputs = {
        REDDIT_SAMPLE_CONSTRUCTION_TABLE: sample_construction,
        REDDIT_SAMPLE_COMPOSITION_TABLE: sample_composition,
        FINBERT_POST_DESCRIPTIVES_TABLE: finbert_descriptives,
        FINBERT_PROCESSING_DIAGNOSTICS_TABLE: processing_diagnostics,
        DAILY_REDDIT_DESCRIPTIVES_TABLE: daily_descriptives,
        SENTIMENT_WEIGHTING_COMPARISON_TABLE: weighting_comparison,
        MARKET_PRICE_COVERAGE_TABLE: market_coverage,
        DAILY_REDDIT_EXTREMES_FILE: daily_extremes,
    }
    for path, table in table_outputs.items():
        write_csv(table, path)

    configure_plots()
    plot_sample_composition(sample_composition)
    plot_finbert_score_distribution(inputs["finbert"])
    plot_finbert_labels(inputs["finbert"])
    plot_daily_sentiment(inputs["daily"])
    plot_daily_attention(inputs["daily"])
    plot_market_levels(inputs["market"])

    hashes_after = validate_input_hashes()
    if hashes_after != hashes_before:
        raise RuntimeError("An immutable input changed while results were generated.")
    validation_summary = build_validation_summary(
        hashes_after,
        inputs["cleaned"],
        inputs["finbert"],
        inputs["daily"],
        inputs["market"],
    )
    write_csv(validation_summary, DESCRIPTIVE_RESULTS_VALIDATION_FILE)

    figure_outputs = [
        REDDIT_SAMPLE_COMPOSITION_FIGURE,
        FINBERT_SENTIMENT_SCORE_FIGURE,
        FINBERT_LABEL_DISTRIBUTION_FIGURE,
        DAILY_REDDIT_SENTIMENT_FIGURE,
        DAILY_REDDIT_ATTENTION_FIGURE,
        MARKET_PRICE_LEVELS_QC_FIGURE,
    ]
    all_outputs = [*table_outputs, DESCRIPTIVE_RESULTS_VALIDATION_FILE, *figure_outputs]
    validate_generated_outputs(all_outputs)

    print("Validated all four immutable empirical inputs and source diagnostics.")
    print(
        f"Reconciled {len(inputs['finbert']):,} posts, "
        f"{inputs['daily']['sentiment'].notna().sum():,} sentiment days, and "
        f"{inputs['market']['index_name'].nunique()} market indices."
    )
    print(
        "Post-weighted mean sentiment: "
        f"{inputs['finbert']['sentiment_score'].mean():.6f}; "
        "populated-day-weighted mean sentiment: "
        f"{inputs['daily']['sentiment'].mean():.6f}."
    )
    print(f"Created {len(table_outputs) + 1} CSV results and {len(figure_outputs)} figures.")
    for output in all_outputs:
        print(f"- {output.relative_to(REPOSITORY_ROOT)}")


if __name__ == "__main__":
    main()
