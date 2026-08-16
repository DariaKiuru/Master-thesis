"""Consolidate frozen Phases 2-7 into the final Phase 8 evidence package."""

from __future__ import annotations

import hashlib
import py_compile
import re
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from config import (  # noqa: E402
    ACTIVE_SCRIPT_INVENTORY_TABLE,
    ALIGNMENT_SAMPLE_SIZES_TABLE,
    DAILY_REDDIT_DESCRIPTIVES_TABLE,
    END_DATE,
    FINAL_METHODOLOGICAL_LIMITATIONS_TABLE,
    FINAL_REGRESSION_TABLE,
    FINAL_REPRODUCIBILITY_AUDIT_FILE,
    FINAL_RESULTS_REVIEW_FILE,
    FINAL_SAMPLE_OVERVIEW_TABLE,
    FINAL_SENTIMENT_RESULTS_SUMMARY_TABLE,
    FINAL_THESIS_OUTPUT_MANIFEST_TABLE,
    FINBERT_POST_DESCRIPTIVES_TABLE,
    FINBERT_REDDIT_FILE,
    GARCH_MODEL_DIAGNOSTICS_FILE,
    GARCH_PARAMETERS_TABLE,
    GARCH_VOLATILITY_DESCRIPTIVES_TABLE,
    HAC_MAX_LAGS,
    MARKET_ALIGNED_LAGGED_FILE,
    MARKET_PRICE_COVERAGE_TABLE,
    MARKET_PRICES_FILE,
    MARKET_RETURNS_GARCH_FILE,
    MARKET_RETURN_DESCRIPTIVES_TABLE,
    MARKET_TICKERS,
    OUTPUTS_DIR,
    REDDIT_SAMPLE_COMPOSITION_TABLE,
    REDDIT_SAMPLE_CONSTRUCTION_TABLE,
    REGRESSION_COEFFICIENTS_LONG_TABLE,
    REGRESSION_MODEL_VALIDATION_FILE,
    REGRESSION_RESULTS_TABLE,
    REGRESSION_SAMPLE_SUPPORT_TABLE,
    REGRESSION_SAMPLE_YEAR_COMPOSITION_TABLE,
    RESEARCH_QUESTION_SUMMARY_TABLE,
    SENTIMENT_COEFFICIENT_COMPARISON_TABLE,
    START_DATE,
    TRADING_DAY_COVERAGE_TABLE,
    TRADING_DAY_MAPPING_RECONCILIATION_TABLE,
)


EXPECTED_FROZEN_HASHES = {
    FINBERT_REDDIT_FILE: (
        "2E4A693558197B8007F81C5D348362140524E3C31A8D31293F86D691DDB9C7FF"
    ),
    MARKET_RETURNS_GARCH_FILE: (
        "FFCF4D30CE1A064B04E726B67273204CF53CDBF1EE5696F1F727251966C425DE"
    ),
    MARKET_ALIGNED_LAGGED_FILE: (
        "1071A9E9BD7E9322F843EE3A60E331B111C70BA386D49E59F2FCC0AC38EA6A86"
    ),
    REGRESSION_COEFFICIENTS_LONG_TABLE: (
        "1853155F3540C55E6AE341515C8382E1DDDEED6D48F0139A75351FC131FF81D5"
    ),
    REGRESSION_RESULTS_TABLE: (
        "C23EB0938B065ADF7E100DD13D83A586E080FF179284FC968AC52058906CCFF7"
    ),
    SENTIMENT_COEFFICIENT_COMPARISON_TABLE: (
        "D66DC053789F72FAA5FF07861727AC4EBAC8922553597ED75FAC5EAD922325AF"
    ),
    REGRESSION_MODEL_VALIDATION_FILE: (
        "C228C401CAA400AAAB779A6A5BD6EFB046747D6F5E0E17EA30ADDC53A69C786A"
    ),
    OUTPUTS_DIR / "figures" / "sentiment_coefficient_comparison.png": (
        "BE7143105EF138E914C89A8087426EA3D2460BFBE30BB1B70CE7E8E7352B89C3"
    ),
}

EXPECTED_MARKETS = ["EURO_STOXX_50", "DAX", "CAC_40", "FTSE_100", "WIG20"]
MARKET_LABELS = {
    "EURO_STOXX_50": "EURO STOXX 50",
    "DAX": "DAX",
    "CAC_40": "CAC 40",
    "FTSE_100": "FTSE 100",
    "WIG20": "WIG20",
}
EXPECTED_REGRESSION_N = {
    "EURO_STOXX_50": 401,
    "DAX": 406,
    "CAC_40": 407,
    "FTSE_100": 397,
    "WIG20": 402,
}
def file_sha256(path: Path) -> str:
    """Return an uppercase SHA-256 digest."""

    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def relative(path: Path) -> str:
    """Return a repository-relative POSIX path."""

    return path.relative_to(REPOSITORY_ROOT).as_posix()


def require_columns(data: pd.DataFrame, columns: set[str], name: str) -> None:
    """Require a minimum CSV schema."""

    missing = columns.difference(data.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")


def read_sources() -> dict[str, pd.DataFrame]:
    """Load only frozen reporting tables; no model is estimated here."""

    def read(path: Path) -> pd.DataFrame:
        return pd.read_csv(path, float_precision="round_trip")

    sources = {
        "sample_construction": read(REDDIT_SAMPLE_CONSTRUCTION_TABLE),
        "sample_composition": read(REDDIT_SAMPLE_COMPOSITION_TABLE),
        "finbert_descriptives": read(FINBERT_POST_DESCRIPTIVES_TABLE),
        "daily_descriptives": read(DAILY_REDDIT_DESCRIPTIVES_TABLE),
        "market_coverage": read(MARKET_PRICE_COVERAGE_TABLE),
        "return_descriptives": read(MARKET_RETURN_DESCRIPTIVES_TABLE),
        "garch_parameters": read(GARCH_PARAMETERS_TABLE),
        "garch_descriptives": read(GARCH_VOLATILITY_DESCRIPTIVES_TABLE),
        "garch_diagnostics": read(GARCH_MODEL_DIAGNOSTICS_FILE),
        "mapping": read(TRADING_DAY_MAPPING_RECONCILIATION_TABLE),
        "trading_coverage": read(TRADING_DAY_COVERAGE_TABLE),
        "alignment_samples": read(ALIGNMENT_SAMPLE_SIZES_TABLE),
        "regression_support": read(REGRESSION_SAMPLE_SUPPORT_TABLE),
        "coefficients": read(REGRESSION_COEFFICIENTS_LONG_TABLE),
        "sentiment_comparison": read(SENTIMENT_COEFFICIENT_COMPARISON_TABLE),
        "year_composition": read(REGRESSION_SAMPLE_YEAR_COMPOSITION_TABLE),
        "regression_validation": read(REGRESSION_MODEL_VALIDATION_FILE),
    }
    require_columns(
        sources["coefficients"],
        {
            "index_name",
            "term",
            "coefficient",
            "hac_standard_error",
            "p_value",
            "ci_lower_95",
            "ci_upper_95",
            "nobs",
            "r_squared",
            "hac_max_lags",
        },
        "regression coefficients",
    )
    require_columns(
        sources["sentiment_comparison"],
        {
            "index_name",
            "coefficient",
            "hac_standard_error",
            "p_value",
            "ci_lower_95",
            "ci_upper_95",
            "nobs",
            "observed_sign",
        },
        "sentiment comparison",
    )
    return sources


def validate_frozen_state() -> dict[str, str]:
    """Stop immediately if any frozen empirical artifact differs."""

    measured: dict[str, str] = {}
    for path, expected in EXPECTED_FROZEN_HASHES.items():
        if not path.exists():
            raise FileNotFoundError(f"Frozen artifact is missing: {path}")
        measured[path.name] = file_sha256(path)
        if measured[path.name] != expected:
            raise ValueError(
                f"Frozen hash changed for {relative(path)}: expected {expected}, "
                f"found {measured[path.name]}."
            )
    return measured


def validate_sources(sources: dict[str, pd.DataFrame]) -> None:
    """Reconcile the approved sample, units, and primary Phase 7 estimates."""

    if START_DATE != "2021-01-01" or END_DATE != "2023-12-31":
        raise ValueError("The final sample period changed.")
    if list(MARKET_TICKERS) != EXPECTED_MARKETS:
        raise ValueError("The approved market order or membership changed.")
    if HAC_MAX_LAGS != 5:
        raise ValueError("The approved HAC maximum lag changed.")

    construction = sources["sample_construction"]
    counts = construction.set_index("stage_or_metric")["count"]
    if int(counts["Phase 3A candidate posts"]) != 3033:
        raise ValueError("Phase 3A candidate count does not reconcile.")
    if int(counts["Final FinBERT-ready posts"]) != 1503:
        raise ValueError("Final Reddit post count does not reconcile.")

    comparison = sources["sentiment_comparison"].set_index("index_name")
    if list(sources["sentiment_comparison"]["index_name"]) != EXPECTED_MARKETS:
        raise ValueError("Sentiment comparison market order changed.")
    if comparison["nobs"].astype(int).to_dict() != EXPECTED_REGRESSION_N:
        raise ValueError("Phase 7 regression sample sizes changed.")
    if int(comparison["coefficient"].lt(0).sum()) != 4:
        raise ValueError("The approved four-negative/one-positive pattern changed.")
    if not (
        comparison["ci_lower_95"].le(0)
        & comparison["ci_upper_95"].ge(0)
    ).all():
        raise ValueError("A frozen lagged-sentiment confidence interval changed.")
    if comparison["p_value"].lt(0.05).any():
        raise ValueError("A frozen lagged-sentiment inference changed.")

    coefficients = sources["coefficients"]
    if not coefficients["hac_max_lags"].eq(5).all():
        raise ValueError("A frozen regression does not use HAC maximum lag 5.")
    validation = sources["regression_validation"]
    if not validation["validation_status"].eq("PASS").all():
        raise ValueError("A frozen regression validation status is not PASS.")
    if not sources["garch_diagnostics"]["converged"].astype(bool).all():
        raise ValueError("A frozen GARCH model is not marked converged.")


def lookup(
    data: pd.DataFrame,
    filters: dict[str, Any],
    value_column: str,
) -> Any:
    """Return one unambiguous table value."""

    mask = pd.Series(True, index=data.index)
    for column, value in filters.items():
        mask &= data[column].astype(str).eq(str(value))
    matches = data.loc[mask, value_column]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one {value_column} value for {filters}, found {len(matches)}."
        )
    return matches.iloc[0]


def build_sample_overview(sources: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build a consolidated table without mixing denominators."""

    rows: list[dict[str, Any]] = []

    def add(
        group: str,
        metric: str,
        value: Any,
        unit: str,
        denominator: str,
        source: Path,
        note: str = "",
        market: str = "",
    ) -> None:
        rows.append(
            {
                "group": group,
                "market": market,
                "metric": metric,
                "value": value,
                "unit": unit,
                "denominator_or_sample": denominator,
                "source_file": relative(source),
                "note": note,
            }
        )

    construction = sources["sample_construction"]
    composition = sources["sample_composition"]
    finbert = sources["finbert_descriptives"]
    daily = sources["daily_descriptives"]
    overall_finbert = finbert.loc[
        finbert["grouping"].eq("overall") & finbert["group"].eq("all posts")
    ].iloc[0]
    overall_daily = daily.loc[daily["period"].astype(str).eq("all")].iloc[0]

    add(
        "Reddit sample",
        "Phase 3A candidates",
        lookup(construction, {"stage_or_metric": "Phase 3A candidate posts"}, "count"),
        "posts",
        "broad validated candidate corpus",
        REDDIT_SAMPLE_CONSTRUCTION_TABLE,
    )
    add(
        "Reddit sample",
        "Final qualifying posts",
        int(overall_finbert["n"]),
        "posts",
        "final pooled three-subreddit sample",
        FINBERT_POST_DESCRIPTIVES_TABLE,
        "Each post receives equal weight.",
    )
    for year in [2021, 2022, 2023]:
        add(
            "Reddit sample by year",
            f"{year} posts",
            lookup(
                composition,
                {"dimension": "year", "category": year},
                "count",
            ),
            "posts",
            "final 1,503-post Reddit sample",
            REDDIT_SAMPLE_COMPOSITION_TABLE,
        )
    for subreddit in ["stocks", "StockMarket", "investing"]:
        add(
            "Reddit sample by subreddit",
            f"r/{subreddit}",
            lookup(
                composition,
                {"dimension": "subreddit", "category": subreddit},
                "count",
            ),
            "posts",
            "final 1,503-post Reddit sample",
            REDDIT_SAMPLE_COMPOSITION_TABLE,
        )
    for label, column in [
        ("Positive labels", "positive_count"),
        ("Neutral labels", "neutral_count"),
        ("Negative labels", "negative_count"),
    ]:
        add(
            "FinBERT labels",
            label,
            int(overall_finbert[column]),
            "posts",
            "argmax descriptive labels in final 1,503-post sample",
            FINBERT_POST_DESCRIPTIVES_TABLE,
        )
    add(
        "Post-level sentiment",
        "Mean sentiment score",
        float(overall_finbert["sentiment_mean"]),
        "positive probability minus negative probability",
        "1,503 equally weighted posts",
        FINBERT_POST_DESCRIPTIVES_TABLE,
    )
    for metric, column, unit, denominator, note in [
        ("Calendar days", "calendar_days", "days", "complete 2021-2023 calendar", ""),
        (
            "Populated sentiment days",
            "sentiment_days",
            "days",
            "complete 1,095-day calendar",
            "Days with at least one qualifying post.",
        ),
        (
            "Zero-post days",
            "zero_post_days",
            "days",
            "complete 1,095-day calendar",
            "Sentiment remains missing; zero posts are not neutral sentiment.",
        ),
        (
            "Populated-day mean sentiment",
            "sentiment_mean",
            "positive probability minus negative probability",
            "504 equally weighted populated calendar days",
            "This differs from the post-weighted mean because the weighting unit differs.",
        ),
    ]:
        add(
            "Calendar-day Reddit series",
            metric,
            overall_daily[column],
            unit,
            denominator,
            DAILY_REDDIT_DESCRIPTIVES_TABLE,
            note,
        )

    coverage = sources["market_coverage"].set_index("index_name")
    returns = sources["return_descriptives"]
    volatility = sources["garch_descriptives"].set_index("index_name")
    alignment = sources["alignment_samples"].set_index("index_name")
    mapping = sources["mapping"].set_index("index_name")
    trading = sources["trading_coverage"].set_index("index_name")
    for market in EXPECTED_MARKETS:
        label = MARKET_LABELS[market]
        log_return_row = returns.loc[
            returns["index_name"].eq(market)
            & returns["variable"].eq("log_return")
        ].iloc[0]
        for group, metric, value, unit, denominator, source, note in [
            (
                "Market data",
                "Price observations",
                int(coverage.loc[market, "number_of_observations"]),
                "trading-date observations",
                "actual market trading calendar",
                MARKET_PRICE_COVERAGE_TABLE,
                "",
            ),
            (
                "Market returns",
                "Finite log returns",
                int(log_return_row["n"]),
                "observations",
                "within-market decimal log returns",
                MARKET_RETURN_DESCRIPTIVES_TABLE,
                "log_return is in decimal units.",
            ),
            (
                "GARCH volatility",
                "Finite conditional standard deviations",
                int(volatility.loc[market, "n"]),
                "observations",
                "constant-mean GARCH(1,1)-Student-t output",
                GARCH_VOLATILITY_DESCRIPTIVES_TABLE,
                "garch_volatility is in percentage-return standard-deviation units.",
            ),
            (
                "Trading-day alignment",
                "Successfully mapped posts",
                int(mapping.loc[market, "total_mapped"]),
                "posts",
                "post-level FinBERT observations mapped for this market",
                TRADING_DAY_MAPPING_RECONCILIATION_TABLE,
                "Mapping is to the same or next actual trading day.",
            ),
            (
                "Trading-day alignment",
                "Zero-attention trading days",
                int(trading.loc[market, "zero_attention_trading_dates"]),
                "trading days",
                f"{int(trading.loc[market, 'total_trading_dates'])} actual trading days",
                TRADING_DAY_COVERAGE_TABLE,
                "Sentiment is missing, not zero, on these trading days.",
            ),
            (
                "Phase 7 eligible sample",
                "Regression observations",
                int(alignment.loc[market, "regression_eligible_rows"]),
                "observations",
                "complete approved Phase 7 variables",
                ALIGNMENT_SAMPLE_SIZES_TABLE,
                "Conditional on observed prior-aligned-trading-day sentiment.",
            ),
        ]:
            add(group, metric, value, unit, denominator, source, note, label)

    return pd.DataFrame(rows)


def build_final_regression_table(sources: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Format the frozen coefficient table without recalculating a model."""

    coefficients = sources["coefficients"].copy()
    terms = [
        "sentiment_lag1",
        "attention_lag1",
        "volatility_lag1",
        "return_lag1",
        "const",
    ]
    rows: list[dict[str, Any]] = []
    for term in terms:
        display_term = "constant" if term == "const" else term
        subset = coefficients.loc[coefficients["term"].eq(term)].set_index(
            "index_name"
        )
        for statistic, column in [
            ("coefficient", "coefficient"),
            ("HAC standard error", "hac_standard_error"),
        ]:
            row: dict[str, Any] = {"term": display_term, "statistic": statistic}
            for market in EXPECTED_MARKETS:
                row[MARKET_LABELS[market]] = subset.loc[market, column]
            row["note"] = ""
            rows.append(row)

    market_level = coefficients.drop_duplicates("index_name").set_index("index_name")
    for statistic, column in [
        ("N", "nobs"),
        ("R-squared", "r_squared"),
        ("HAC maximum lag", "hac_max_lags"),
    ]:
        row = {"term": "model", "statistic": statistic}
        for market in EXPECTED_MARKETS:
            value = market_level.loc[market, column]
            row[MARKET_LABELS[market]] = int(value) if statistic != "R-squared" else value
        row["note"] = (
            "OLS with HAC/Newey-West standard errors; maximum lag = 5."
            if statistic == "HAC maximum lag"
            else ""
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_sentiment_summary(sources: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Create the compact primary-result table from frozen Phase 7 values."""

    comparison = sources["sentiment_comparison"]
    rows = []
    for market in EXPECTED_MARKETS:
        source = comparison.loc[comparison["index_name"].eq(market)].iloc[0]
        ci_includes_zero = bool(
            source["ci_lower_95"] <= 0 <= source["ci_upper_95"]
        )
        observed_sign = "negative" if source["coefficient"] < 0 else "positive"
        rows.append(
            {
                "market": MARKET_LABELS[market],
                "N": int(source["nobs"]),
                "sentiment_coefficient": source["coefficient"],
                "HAC_SE": source["hac_standard_error"],
                "p_value": source["p_value"],
                "CI_lower": source["ci_lower_95"],
                "CI_upper": source["ci_upper_95"],
                "observed_sign": observed_sign,
                "H1_predicted_sign": "negative",
                "CI_includes_zero": ci_includes_zero,
                "summary_interpretation": (
                    f"{observed_sign} direction, statistically imprecise"
                ),
            }
        )
    return pd.DataFrame(rows)


def build_research_question_summary(
    sentiment_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Synthesize RQ1, RQ2, and H1 after validating the frozen evidence."""

    negative_count = int(sentiment_summary["sentiment_coefficient"].lt(0).sum())
    if negative_count != 4 or not sentiment_summary["CI_includes_zero"].all():
        raise ValueError("The approved RQ/H1 evidence pattern changed.")
    return pd.DataFrame(
        [
            {
                "item": "RQ1",
                "question_or_hypothesis": (
                    "Is Reddit-based investor sentiment significantly associated "
                    "with subsequent market volatility?"
                ),
                "evidence": (
                    "All five lagged-sentiment 95% HAC confidence intervals include "
                    "zero and all five p-values exceed 0.05."
                ),
                "conclusion": (
                    "The frozen market-specific OLS-HAC models do not provide "
                    "statistically significant evidence that lagged Reddit sentiment "
                    "is associated with subsequent GARCH conditional volatility in "
                    "any of the five markets."
                ),
                "inference_scope": "Associational evidence only.",
            },
            {
                "item": "RQ2",
                "question_or_hypothesis": (
                    "Does the sentiment-volatility relationship differ across the "
                    "five markets?"
                ),
                "evidence": (
                    f"Point estimates include {negative_count} negative estimates "
                    "and one positive WIG20 estimate; all intervals include zero and "
                    "overlap substantially."
                ),
                "conclusion": (
                    "Point estimates differ across markets, but the results do not "
                    "provide strong evidence of systematic cross-market differences "
                    "in the sentiment-volatility relationship."
                ),
                "inference_scope": (
                    "No formal coefficient-equality test was part of the frozen design."
                ),
            },
            {
                "item": "H1",
                "question_or_hypothesis": (
                    "More negative Reddit sentiment is associated with higher "
                    "subsequent volatility (predicted negative coefficient)."
                ),
                "evidence": (
                    f"{negative_count} of five coefficients have the hypothesized "
                    "negative sign, but none is statistically significant."
                ),
                "conclusion": (
                    "The results are directionally consistent with H1 in four markets "
                    "but do not provide statistically significant support for H1 overall."
                ),
                "inference_scope": "H1 is neither proven nor conclusively rejected.",
            },
        ]
    )


def build_limitations(sources: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Create evidence-based limitations from observed sample structure."""

    composition = sources["sample_composition"]
    daily = sources["daily_descriptives"]
    alignment = sources["alignment_samples"]
    support = sources["regression_support"]
    years = sources["year_composition"]
    share_2022 = float(
        lookup(
            composition,
            {"dimension": "year", "category": 2022},
            "percentage",
        )
    )
    zero_days = int(
        daily.loc[daily["period"].astype(str).eq("all"), "zero_post_days"].iloc[0]
    )
    min_n = int(alignment["regression_eligible_rows"].min())
    max_n = int(alignment["regression_eligible_rows"].max())
    retention_min = float(
        alignment["percentage_finite_volatility_rows_retained"].min()
    )
    retention_max = float(
        alignment["percentage_finite_volatility_rows_retained"].max()
    )
    attention_min = int(support["attention_lag1_minimum"].min())
    year_min = float(
        years.loc[years["year"].eq(2022), "share_of_market_regression_sample"].min()
        * 100
    )
    year_max = float(
        years.loc[years["year"].eq(2022), "share_of_market_regression_sample"].max()
        * 100
    )
    return pd.DataFrame(
        [
            {
                "limitation_id": 1,
                "limitation": "Reddit sample scope",
                "evidence": (
                    "The sample pools r/investing, r/stocks, and r/StockMarket."
                ),
                "implication": (
                    "Sentiment reflects selected finance-oriented Reddit communities, "
                    "not all investors."
                ),
            },
            {
                "limitation_id": 2,
                "limitation": "Temporal concentration",
                "evidence": f"{share_2022:.1f}% of final Reddit posts occur in 2022.",
                "implication": "The corpus is concentrated in the crisis's central year.",
            },
            {
                "limitation_id": 3,
                "limitation": "Calendar sparsity",
                "evidence": (
                    f"{zero_days} of 1,095 calendar days have no qualifying posts."
                ),
                "implication": "Zero-post-day sentiment remains missing, not neutral.",
            },
            {
                "limitation_id": 4,
                "limitation": "Regression-sample selection",
                "evidence": (
                    f"Phase 7 uses {min_n}-{max_n} observations per market, retaining "
                    f"{retention_min:.1f}%-{retention_max:.1f}% of finite-volatility rows."
                ),
                "implication": (
                    "Inference pertains to the approved eligible samples rather than "
                    "every finite-volatility trading day."
                ),
            },
            {
                "limitation_id": 5,
                "limitation": "Conditional observability",
                "evidence": (
                    "Eligibility requires nonmissing sentiment on the preceding aligned "
                    "trading day."
                ),
                "implication": (
                    "The regression sample is conditional on qualifying Reddit "
                    "sentiment being observed."
                ),
            },
            {
                "limitation_id": 6,
                "limitation": "Attention support",
                "evidence": (
                    f"The minimum attention_lag1 in every eligible sample is "
                    f"{attention_min}; no eligible row has zero prior-day attention."
                ),
                "implication": (
                    "Attention measures discussion intensity conditional on discussion "
                    "occurring, not zero versus positive discussion."
                ),
            },
            {
                "limitation_id": 7,
                "limitation": "Regression-year concentration",
                "evidence": (
                    f"2022 supplies {year_min:.1f}%-{year_max:.1f}% of each market's "
                    "estimation observations."
                ),
                "implication": "The estimation samples are concentrated in 2022.",
            },
            {
                "limitation_id": 8,
                "limitation": "Observational design",
                "evidence": "Predictors are lagged within the market trading calendar.",
                "implication": "Lagging variables does not establish structural causation.",
            },
            {
                "limitation_id": 9,
                "limitation": "Sentiment measurement",
                "evidence": (
                    "ProsusAI/finbert is applied as a pretrained model without "
                    "thesis-specific fine-tuning."
                ),
                "implication": "The sentiment measure inherits the model's domain limits.",
            },
            {
                "limitation_id": 10,
                "limitation": "Market-specific estimates",
                "evidence": (
                    "Point estimates differ, but no coefficient-equality test was part "
                    "of the frozen design."
                ),
                "implication": (
                    "The observed differences should not be described as formally "
                    "established heterogeneity."
                ),
            },
        ]
    )


def build_script_inventory() -> pd.DataFrame:
    """Document the active numbered workflow in execution order."""

    rows = [
        ("01_download_market_data.py", "Phase 2", "Remote Yahoo Finance and archived/live Stooq responses", "data/processed/market_prices.csv", "No - network collection", "Yes", "Yes", "No; verify the frozen output instead"),
        ("02_extract_reddit.py", "Phase 3A", "Arctic Shift historical Reddit API", "data/raw/reddit/reddit_posts_2021_2023_raw.csv", "No - network collection", "Yes", "Yes", "No; verify the frozen corpus instead"),
        ("03_inspect_reddit_candidates.py", "Phase 3A audit", "data/raw/reddit/reddit_posts_2021_2023_raw.csv", "outputs/diagnostics/reddit_candidate_corpus_profile.csv", "Yes", "No", "Yes", "Only when reproducing the frozen candidate audit"),
        ("04_validate_reddit_relevance.py", "Phase 3B.2", "Frozen Phase 3A candidate corpus", "outputs/diagnostics/reddit_relevance_filter_dry_run.csv", "Yes", "No", "Yes", "Only when reproducing the historical rule-validation audit"),
        ("05_compare_reddit_relevance_rules.py", "Phase 3B.3", "Frozen candidates and Phase 3B.2 dry run", "outputs/diagnostics/reddit_relevance_rule_comparison.csv", "Yes", "No", "Yes", "Only when reproducing the historical rule comparison"),
        ("06_clean_reddit.py", "Phase 3B final", "data/raw/reddit/reddit_posts_2021_2023_raw.csv", "data/processed/reddit_posts_cleaned.csv", "Yes", "No", "Yes", "No; verify the frozen cleaned corpus"),
        ("07_score_finbert.py", "Phase 4A", "data/processed/reddit_posts_cleaned.csv", "data/processed/reddit_posts_finbert.csv", "No - computationally expensive", "Conditional first-time model download", "Yes", "No; validated output is skipped when present"),
        ("08_build_daily_sentiment.py", "Phase 4B", "data/processed/reddit_posts_finbert.csv", "data/processed/daily_reddit_sentiment.csv", "Yes", "No", "Yes", "Only for an explicit frozen-stage reproduction"),
        ("09_build_descriptive_results.py", "Phase 4C", "Frozen market, cleaned Reddit, FinBERT, and daily files", "Phase 4C tables, diagnostics, and figures under outputs/", "Yes", "No", "Yes", "Only for an explicit frozen-stage reproduction"),
        ("10_build_market_volatility.py", "Phase 5", "data/processed/market_prices.csv", "data/processed/market_returns_garch.csv", "Yes - moderate GARCH estimation", "No", "Yes", "No; verify the frozen Phase 5 hash"),
        ("11_build_trading_day_alignment.py", "Phase 6", "Post-level FinBERT scores and frozen Phase 5 market panel", "data/processed/market_aligned_lagged.csv", "Yes", "No", "Yes", "No; verify the frozen Phase 6 hash"),
        ("12_run_hac_regressions.py", "Phase 7", "data/processed/market_aligned_lagged.csv", "outputs/tables/regression_coefficients_long.csv", "Yes", "No", "Yes", "No; verify the frozen Phase 7 hashes"),
        ("13_build_final_thesis_outputs.py", "Phase 8", "Frozen reporting tables and diagnostics from Phases 2-7", "Final synthesis tables, manifest, audit, and review note", "Yes", "No", "No - reporting only", "Yes, to reproduce the final reporting package"),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "script",
            "phase",
            "primary_input",
            "primary_output",
            "deterministic_and_inexpensive",
            "depends_on_network_access",
            "frozen",
            "normally_rerun",
        ],
    )


MAIN_TEXT_FILES = {
    "tables/reddit_sample_construction.csv",
    "tables/finbert_post_sentiment_descriptives.csv",
    "tables/daily_reddit_descriptives.csv",
    "tables/market_return_descriptives.csv",
    "tables/garch_parameters.csv",
    "tables/garch_volatility_descriptives.csv",
    "tables/regression_results.csv",
    "tables/final_sample_overview.csv",
    "tables/final_regression_table.csv",
    "tables/final_sentiment_results_summary.csv",
    "tables/research_question_summary.csv",
    "tables/final_methodological_limitations.csv",
    "figures/reddit_sample_composition.png",
    "figures/finbert_label_distribution.png",
    "figures/daily_reddit_sentiment.png",
    "figures/garch_conditional_volatility.png",
    "figures/sentiment_coefficient_comparison.png",
}

REDUNDANT_INTERNAL_FILES = {
    "diagnostics/daily_reddit_extremes.csv",
    "diagnostics/finbert_development_sample.csv",
    "diagnostics/finbert_inference_benchmark.csv",
    "diagnostics/finbert_review_sample.csv",
    "diagnostics/reddit_combined_query_differences.csv",
    "diagnostics/reddit_filter_excluded_review.csv",
    "diagnostics/reddit_filter_included_review.csv",
    "diagnostics/reddit_relevance_filter_dry_run.csv",
    "diagnostics/reddit_relevance_review_sample.csv",
    "diagnostics/reddit_rule_newly_excluded_review.csv",
    "diagnostics/reddit_rule_newly_included_review.csv",
}

PHASE_FILES = {
    "Phase 2": {
        "diagnostics/market_data_summary.csv",
    },
    "Phase 3A": {
        "diagnostics/reddit_candidate_corpus_profile.csv",
        "diagnostics/reddit_combined_query_differences.csv",
        "diagnostics/reddit_combined_query_validation.csv",
        "diagnostics/reddit_extraction_summary.csv",
        "diagnostics/reddit_query_summary.csv",
        "diagnostics/reddit_relevance_review_sample.csv",
    },
    "Phase 3B": {
        "diagnostics/reddit_cleaning_summary.csv",
        "diagnostics/reddit_filter_excluded_review.csv",
        "diagnostics/reddit_filter_included_review.csv",
        "diagnostics/reddit_language_validation.csv",
        "diagnostics/reddit_relevance_filter_dry_run.csv",
        "diagnostics/reddit_relevance_rule_comparison.csv",
        "diagnostics/reddit_rule_newly_excluded_review.csv",
        "diagnostics/reddit_rule_newly_included_review.csv",
    },
    "Phase 4A": {
        "diagnostics/finbert_class_distribution.csv",
        "diagnostics/finbert_development_sample.csv",
        "diagnostics/finbert_inference_benchmark.csv",
        "diagnostics/finbert_review_sample.csv",
        "diagnostics/finbert_sentiment_summary.csv",
    },
    "Phase 4B": {"diagnostics/daily_reddit_sentiment_summary.csv"},
    "Phase 4C": {
        "diagnostics/daily_reddit_extremes.csv",
        "diagnostics/descriptive_results_validation.csv",
        "tables/daily_reddit_descriptives.csv",
        "tables/finbert_post_sentiment_descriptives.csv",
        "tables/finbert_processing_diagnostics.csv",
        "tables/market_price_coverage.csv",
        "tables/reddit_sample_composition.csv",
        "tables/reddit_sample_construction.csv",
        "tables/sentiment_weighting_comparison.csv",
        "figures/daily_reddit_attention.png",
        "figures/daily_reddit_sentiment.png",
        "figures/finbert_label_distribution.png",
        "figures/finbert_sentiment_score_distribution.png",
        "figures/market_price_levels_qc.png",
        "figures/reddit_sample_composition.png",
    },
    "Phase 5": {
        "diagnostics/garch_model_diagnostics.csv",
        "tables/garch_parameters.csv",
        "tables/garch_volatility_descriptives.csv",
        "tables/market_return_descriptives.csv",
        "figures/garch_conditional_volatility.png",
        "figures/market_log_returns.png",
    },
    "Phase 6": {
        "diagnostics/alignment_timing_review_sample.csv",
        "diagnostics/alignment_weighting_validation.csv",
        "diagnostics/reddit_trading_day_mapping.csv",
        "diagnostics/sentiment_missingness_transitions.csv",
        "diagnostics/trading_day_alignment_validation.csv",
        "tables/aligned_reddit_descriptives.csv",
        "tables/alignment_sample_sizes.csv",
        "tables/regression_sample_support.csv",
        "tables/trading_day_coverage.csv",
        "tables/trading_day_mapping_reconciliation.csv",
        "tables/yearly_regression_support.csv",
        "figures/market_aligned_attention.png",
    },
    "Phase 7": {
        "diagnostics/regression_model_validation.csv",
        "tables/regression_coefficients_long.csv",
        "tables/regression_results.csv",
        "tables/regression_sample_descriptives.csv",
        "tables/regression_sample_year_composition.csv",
        "tables/sentiment_coefficient_comparison.csv",
        "figures/sentiment_coefficient_comparison.png",
    },
}


def phase_for(output_relative: str) -> str:
    """Assign an output to its producing phase."""

    if output_relative.startswith("tables/final_") or output_relative in {
        "tables/research_question_summary.csv",
        "tables/active_script_inventory.csv",
        "diagnostics/final_reproducibility_audit.csv",
        "final_results_review.md",
    }:
        return "Phase 8"
    for phase, names in PHASE_FILES.items():
        if output_relative in names:
            return phase
    raise ValueError(f"Output is missing a phase classification: {output_relative}")


def topic_for(name: str) -> str:
    """Provide a compact topic label for the manifest."""

    lowered = name.lower()
    for token, topic in [
        ("regression", "OLS-HAC regression"),
        ("sentiment", "Reddit sentiment"),
        ("attention", "Reddit attention"),
        ("garch", "GARCH conditional volatility"),
        ("market", "equity-market data"),
        ("reddit", "Reddit sample"),
        ("alignment", "trading-day alignment"),
        ("mapping", "trading-day mapping"),
        ("script", "active pipeline"),
        ("reproducibility", "reproducibility"),
        ("limitation", "methodological limitations"),
        ("research_question", "RQ1, RQ2, and H1"),
    ]:
        if token in lowered:
            return topic
    return "final empirical workflow"


def units_for(name: str) -> str:
    """Describe units without confusing variance and standard deviation."""

    lowered = name.lower()
    if "garch" in lowered or "volatility" in lowered:
        return "percentage-return conditional standard-deviation units where applicable"
    if "return" in lowered:
        return "decimal log returns and/or percentage-return units as explicitly labelled"
    if "attention" in lowered or "count" in lowered or "sample" in lowered:
        return "counts, shares, or explicitly labelled variables"
    if "sentiment" in lowered:
        return "positive probability minus negative probability where applicable"
    return "units stated in the artifact or not applicable"


def sample_for(phase: str, name: str) -> str:
    """Describe the relevant sample denominator."""

    if phase in {"Phase 3A", "Phase 3B"}:
        return "Reddit candidate or final post sample, 2021-01-01 to 2023-12-31"
    if phase in {"Phase 4A", "Phase 4B", "Phase 4C"}:
        return "1,503 final posts or complete 1,095-day calendar, as labelled"
    if phase == "Phase 5":
        return "five market-specific trading calendars, 2021-2023"
    if phase == "Phase 6":
        return "post-level mapping to each market's actual trading calendar"
    if phase == "Phase 7":
        return "market-specific eligible samples (N = 397-407)"
    if phase == "Phase 8":
        return "frozen approved evidence from Phases 2-7"
    return "sample stated in the artifact"


def build_manifest() -> pd.DataFrame:
    """Inventory and classify every retained output artifact."""

    paths = sorted(
        path
        for path in OUTPUTS_DIR.rglob("*")
        if path.is_file() and path.name != ".gitkeep"
    )
    rows = []
    for path in paths:
        output_relative = path.relative_to(OUTPUTS_DIR).as_posix()
        phase = phase_for(output_relative)
        if output_relative in MAIN_TEXT_FILES:
            classification = "MAIN THESIS"
            location = "main_text"
            status = (
                "authoritative frozen source"
                if output_relative == "tables/regression_results.csv"
                else "recommended"
            )
        elif output_relative in REDUNDANT_INTERNAL_FILES:
            classification = "REDUNDANT / INTERNAL QC"
            location = "technical_qc"
            status = "retained internal validation artifact"
        else:
            classification = "APPENDIX / TECHNICAL DIAGNOSTIC"
            location = (
                "technical_qc"
                if output_relative.startswith("diagnostics/")
                or output_relative in {
                    "tables/active_script_inventory.csv",
                    "tables/final_thesis_output_manifest.csv",
                }
                else "appendix"
            )
            status = "retained supporting artifact"
        artifact_type = (
            "figure"
            if path.suffix.lower() == ".png"
            else "researcher synthesis note"
            if path.suffix.lower() == ".md"
            else "table or diagnostic CSV"
        )
        rows.append(
            {
                "filename": f"outputs/{output_relative}",
                "phase": phase,
                "type": artifact_type,
                "classification": classification,
                "recommended_location": location,
                "purpose": path.stem.replace("_", " "),
                "main_variable_or_topic": topic_for(path.name),
                "units": units_for(path.name),
                "sample_definition": sample_for(phase, path.name),
                "status": status,
            }
        )
    return pd.DataFrame(rows)


def png_metadata(path: Path) -> tuple[int, int, float | None]:
    """Read PNG pixel dimensions and physical-resolution metadata."""

    with path.open("rb") as image_file:
        if image_file.read(8) != b"\x89PNG\r\n\x1a\n":
            raise ValueError(f"Not a PNG file: {path}")
        width = height = 0
        dpi: float | None = None
        while True:
            length_bytes = image_file.read(4)
            if not length_bytes:
                break
            length = struct.unpack(">I", length_bytes)[0]
            chunk_type = image_file.read(4)
            data = image_file.read(length)
            image_file.read(4)
            if chunk_type == b"IHDR":
                width, height = struct.unpack(">II", data[:8])
            elif chunk_type == b"pHYs":
                x_pixels_per_meter, _, unit = struct.unpack(">IIB", data)
                if unit == 1:
                    dpi = x_pixels_per_meter * 0.0254
            elif chunk_type == b"IEND":
                break
    return width, height, dpi


def git_tracked_files() -> list[str]:
    """Return tracked repository paths without modifying Git state."""

    result = subprocess.run(
        ["git", "ls-files"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def build_reproducibility_audit(
    measured_hashes: dict[str, str],
    sources: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Run inexpensive repository, path, figure, and hygiene checks."""

    rows: list[dict[str, Any]] = []

    def record(
        area: str,
        check: str,
        passed: bool,
        measured: Any,
        expected: Any,
        details: str = "",
    ) -> None:
        rows.append(
            {
                "area": area,
                "check": check,
                "status": "PASS" if passed else "FAIL",
                "measured_value": measured,
                "expected_value": expected,
                "details": details,
            }
        )
        if not passed:
            raise RuntimeError(f"Reproducibility audit failed: {area} / {check}")

    for path, expected in EXPECTED_FROZEN_HASHES.items():
        measured = measured_hashes[path.name]
        record("frozen hashes", relative(path), measured == expected, measured, expected)

    required_inputs = [
        MARKET_PRICES_FILE,
        FINBERT_REDDIT_FILE,
        MARKET_RETURNS_GARCH_FILE,
        MARKET_ALIGNED_LAGGED_FILE,
        REDDIT_SAMPLE_CONSTRUCTION_TABLE,
        REGRESSION_COEFFICIENTS_LONG_TABLE,
        REGRESSION_RESULTS_TABLE,
    ]
    for path in required_inputs:
        record("paths", relative(path), path.exists(), path.exists(), True)

    market_names = set(sources["market_coverage"]["index_name"])
    record(
        "market naming",
        "exact approved market keys",
        market_names == set(EXPECTED_MARKETS),
        "|".join(sorted(market_names)),
        "|".join(sorted(EXPECTED_MARKETS)),
    )
    wig20 = sources["market_coverage"].set_index("index_name").loc["WIG20"]
    wig_ok = (
        str(wig20["source_symbol"]) == "wig20"
        and str(wig20["ticker"]) == "WIG20"
        and str(wig20["data_source"]) == "Stooq"
        and "Internet Archive" in str(wig20["retrieval_method"])
    )
    record(
        "market naming",
        "WIG20 provenance",
        wig_ok,
        f"{wig20['source_symbol']}|{wig20['ticker']}|{wig20['data_source']}|{wig20['retrieval_method']}",
        "wig20|WIG20|Stooq|Internet Archive snapshot of Stooq",
    )

    active_scripts = sorted((REPOSITORY_ROOT / "src").glob("[0-9][0-9]_*.py"))
    compile_failures = []
    for script in active_scripts:
        try:
            py_compile.compile(str(script), doraise=True)
        except py_compile.PyCompileError as error:
            compile_failures.append(f"{script.name}: {error}")
    record(
        "compilation",
        "all active numbered scripts compile",
        not compile_failures and len(active_scripts) == 13,
        len(active_scripts) - len(compile_failures),
        13,
        " | ".join(compile_failures),
    )

    requirements = {
        line.split("==", 1)[0].strip().lower()
        for line in (REPOSITORY_ROOT / "requirements.txt").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    required_packages = {
        "arch",
        "langdetect",
        "matplotlib",
        "numpy",
        "pandas",
        "requests",
        "statsmodels",
        "torch",
        "transformers",
        "yfinance",
    }
    record(
        "requirements",
        "all imported third-party packages declared",
        required_packages.issubset(requirements),
        "|".join(sorted(requirements)),
        "|".join(sorted(required_packages)),
    )

    readme = (REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8")
    agents = (REPOSITORY_ROOT / "AGENTS.md").read_text(encoding="utf-8")
    documentation_ok = (
        "13_build_final_thesis_outputs.py" in readme
        and "Phase 8" in readme
        and "Phase 8" in agents
        and "HAC/Newey-West" in readme
    )
    record(
        "documentation",
        "README and AGENTS describe completed Phase 8 workflow",
        documentation_ok,
        documentation_ok,
        True,
    )

    empirical_scripts = [
        path for path in active_scripts if path.name != "13_build_final_thesis_outputs.py"
    ]
    source_text = "\n".join(
        path.read_text(encoding="utf-8") for path in empirical_scripts
    )
    excluded_patterns = [
        r"\bwavelet\b",
        r"\bevent[- ]study\b",
        r"\bgranger\b",
        r"\bfama[- ]french\b",
        r"\bgarch[- ]x\b",
        r"\bsvm\b",
        r"machine[- ]learning market[- ]prediction",
        r"finbert fine[- ]tuning",
    ]
    active_hits = [
        pattern
        for pattern in excluded_patterns
        if re.search(pattern, source_text, flags=re.IGNORECASE)
    ]
    record(
        "archive separation",
        "excluded empirical methods absent from scripts 01-12",
        not active_hits,
        "|".join(active_hits),
        "none",
        "Historical files remain under archive/.",
    )
    record(
        "archive separation",
        "archive directory retained",
        (REPOSITORY_ROOT / "archive").is_dir(),
        (REPOSITORY_ROOT / "archive").is_dir(),
        True,
    )

    gitignore = (REPOSITORY_ROOT / ".gitignore").read_text(encoding="utf-8")
    ignore_tokens = [".venv/", "__pycache__/", "*.py[cod]", ".env", "checkpoints/"]
    missing_ignores = [token for token in ignore_tokens if token not in gitignore]
    record(
        "Git hygiene",
        "environment, cache, secret, and checkpoint ignores",
        not missing_ignores,
        "|".join(missing_ignores),
        "none missing",
    )
    tracked = git_tracked_files()
    suspicious_names = [
        name
        for name in tracked
        if Path(name).name.lower() in {".env", "credentials.json", "secrets.json"}
        or Path(name).suffix.lower() in {".pem", ".p12", ".pfx", ".key"}
    ]
    record(
        "Git hygiene",
        "no obviously secret-bearing filenames tracked",
        not suspicious_names,
        "|".join(suspicious_names),
        "none",
    )
    normalized_tracked = [(name, name.replace("\\", "/")) for name in tracked]
    unwanted_tracked = [
        name
        for name, normalized in normalized_tracked
        if "/__pycache__/" in f"/{normalized}"
        or name.endswith((".pyc", ".pyo"))
        or normalized.startswith((".venv/", "venv/", "checkpoints/"))
    ]
    record(
        "Git hygiene",
        "no environment, cache, or checkpoint artifacts tracked",
        not unwanted_tracked,
        "|".join(unwanted_tracked),
        "none",
    )

    figures = sorted((OUTPUTS_DIR / "figures").glob("*.png"))
    for figure in figures:
        width, height, dpi = png_metadata(figure)
        resolution_ok = width >= 2000 and height >= 1200
        dpi_ok = dpi is not None and abs(dpi - 300) < 1
        record(
            "figure quality",
            figure.name,
            resolution_ok and dpi_ok,
            f"{width}x{height}; {dpi:.1f} DPI" if dpi is not None else f"{width}x{height}; no DPI",
            "at least 2000x1200 pixels; approximately 300 DPI",
            "Visual clipping and label review completed during Phase 8.",
        )

    record(
        "statistical interpretation",
        "all sentiment confidence intervals include zero",
        bool(sources["sentiment_comparison"]["ci_excludes_zero"].eq(False).all()),
        int(sources["sentiment_comparison"]["ci_excludes_zero"].sum()),
        0,
    )
    record(
        "statistical interpretation",
        "approved four-negative/one-positive sign pattern",
        int(sources["sentiment_comparison"]["coefficient"].lt(0).sum()) == 4,
        int(sources["sentiment_comparison"]["coefficient"].lt(0).sum()),
        4,
    )
    return pd.DataFrame(rows)


def format_sentiment_result(row: pd.Series) -> str:
    """Format one exact approved primary result for Markdown."""

    return (
        f"| {row['market']} | {int(row['N'])} | "
        f"{row['sentiment_coefficient']:.9f} | {row['HAC_SE']:.9f} | "
        f"{row['p_value']:.9f} | [{row['CI_lower']:.9f}, "
        f"{row['CI_upper']:.9f}] |"
    )


def build_review_note(
    sources: dict[str, pd.DataFrame],
    sentiment_summary: pd.DataFrame,
) -> str:
    """Generate the researcher-facing synthesis note from frozen sources."""

    finbert = sources["finbert_descriptives"]
    overall = finbert.loc[
        finbert["grouping"].eq("overall") & finbert["group"].eq("all posts")
    ].iloc[0]
    daily = sources["daily_descriptives"].loc[
        sources["daily_descriptives"]["period"].astype(str).eq("all")
    ].iloc[0]
    garch = sources["garch_descriptives"].set_index("index_name")
    parameters = sources["garch_parameters"].set_index("index_name")
    mapping = sources["mapping"]
    trading = sources["trading_coverage"]
    alignment = sources["alignment_samples"]
    years = sources["year_composition"]
    coefficients = sources["coefficients"]
    attention = coefficients.loc[coefficients["term"].eq("attention_lag1")]
    persistence = coefficients.loc[coefficients["term"].eq("volatility_lag1")]

    result_rows = "\n".join(
        format_sentiment_result(row) for _, row in sentiment_summary.iterrows()
    )
    garch_persistence = ", ".join(
        f"{MARKET_LABELS[market]} {parameters.loc[market, 'alpha_plus_beta']:.3f}"
        for market in EXPECTED_MARKETS
    )
    return f"""# Final results review

This is a researcher-facing synthesis of the frozen empirical evidence. It is
not a finished thesis chapter and introduces no additional empirical model.

## 1. Sample construction

The validated Phase 3A corpus contains 3,033 candidate posts. The final pooled
sample contains 1,503 equally weighted posts from r/investing, r/stocks, and
r/StockMarket. Of these, 1,197 ({1197 / 1503 * 100:.1f}%) occur in 2022, compared
with 83 in 2021 and 223 in 2023. The subreddit counts are 772 for r/stocks, 433
for r/StockMarket, and 298 for r/investing.

## 2. Sentiment descriptives

FinBERT's descriptive argmax labels comprise {int(overall['positive_count'])}
positive, {int(overall['neutral_count'])} neutral, and
{int(overall['negative_count'])} negative posts. Neutral labels therefore
dominate. The equal-post-weight mean sentiment score is
{overall['sentiment_mean']:.6f}. The complete 1,095-day calendar contains
{int(daily['sentiment_days'])} populated sentiment days and
{int(daily['zero_post_days'])} zero-post days. Sentiment remains missing on
zero-post days. The mean across populated calendar days is
{daily['sentiment_mean']:.6f}; it differs from the post-level mean because it
weights populated days rather than individual posts.

## 3. Market returns and GARCH

WIG20 has the largest raw log-return standard deviation and the highest average
conditional volatility ({garch.loc['WIG20', 'mean']:.3f} percentage-return
standard-deviation units). FTSE 100 has the lowest corresponding raw return
variability and average conditional volatility ({garch.loc['FTSE_100', 'mean']:.3f}).
The largest conditional-volatility readings occur in late February or March
2022. All five constant-mean GARCH(1,1)-Student-t models converged. Estimated
GARCH persistence (alpha + beta) is {garch_persistence}; persistence is high,
especially for WIG20, DAX, EURO STOXX 50, and CAC 40.

## 4. Trading-day alignment and missingness

Regression alignment starts from the 1,503 post-level FinBERT observations, not
the calendar-day sentiment means. All posts map successfully to the same or next
actual trading date for every market ({int(mapping['total_mapped'].min()):,} of
1,503 in each case). Market-calendar differences create some different mapping
dates. Across markets, {trading['zero_attention_share'].min() * 100:.1f}% to
{trading['zero_attention_share'].max() * 100:.1f}% of trading days have zero
attention; aligned sentiment remains missing on those days. No sentiment
imputation is used. Phase 7 retains
{alignment['percentage_finite_volatility_rows_retained'].min():.1f}% to
{alignment['percentage_finite_volatility_rows_retained'].max():.1f}% of finite
volatility observations.

## 5. Regression results

The approved specification estimates five separate intercept-including OLS
models with HAC/Newey-West standard errors (maximum lag 5). `garch_volatility`
is the dependent variable; regressors are `sentiment_lag1`, `attention_lag1`,
`volatility_lag1`, and decimal `return_lag1`.

| Market | N | Sentiment coefficient | HAC SE | p-value | 95% CI |
|---|---:|---:|---:|---:|---:|
{result_rows}

Four point estimates are negative, as predicted by H1, while WIG20's estimate
is positive. Every 95% confidence interval includes zero, and none of the five
sentiment coefficients is statistically significant. Lagged volatility is
strongly positive in every market (coefficients
{persistence['coefficient'].min():.3f}-{persistence['coefficient'].max():.3f})
and statistically precise. Attention is positive in all five models
({attention['coefficient'].min():.4f}-{attention['coefficient'].max():.4f}), but
its interpretation is conditional on positive prior-day discussion because
every eligible observation has `attention_lag1 > 0`. Approximately
{years.loc[years['year'].eq(2022), 'share_of_market_regression_sample'].min() * 100:.1f}%
to {years.loc[years['year'].eq(2022), 'share_of_market_regression_sample'].max() * 100:.1f}%
of each estimation sample comes from 2022.

## 6. RQ1

The frozen market-specific OLS-HAC models do not provide statistically
significant evidence that lagged Reddit sentiment is associated with subsequent
GARCH conditional volatility in any of the five markets under the approved
specification.

## 7. RQ2

Point estimates differ across markets, including four negative estimates and
one positive WIG20 estimate, but all confidence intervals include zero and
overlap substantially. The results do not provide strong evidence of systematic
cross-market heterogeneity. No formal coefficient-equality test was part of the
frozen design.

## 8. H1

Four of five coefficients have the hypothesized negative sign, but none is
statistically significant. The results are directionally consistent with H1 in
four markets but do not provide statistically significant support for H1
overall.

## 9. Main limitations

- Sentiment represents three selected finance-oriented Reddit communities, not
  all investors.
- The Reddit corpus and regression samples are concentrated in 2022.
- There are 591 zero-post calendar days, and eligible regression samples are
  conditional on observed prior-aligned-trading-day sentiment.
- Attention varies only over positive values in the eligible samples.
- FinBERT is used pretrained without thesis-specific fine-tuning.
- Lagging predictors does not establish structural causation.
- Cross-market point estimates are not a formal coefficient-equality test.

## 10. Main reproducibility conclusions

The frozen canonical hashes match the approved checkpoints. Active scripts
compile, required inputs and configured output paths exist, market names and
units are consistent, archived experiments remain separate from the numbered
pipeline, and Git ignore rules cover environments, caches, local secrets, and
checkpoints. The final output inventory is in
`outputs/tables/final_thesis_output_manifest.csv`; detailed audit checks are in
`outputs/diagnostics/final_reproducibility_audit.csv`.

All statements above are associational. No causal claim, alternative
specification, robustness search, sentiment imputation, or new sentiment index
is used.
"""


def write_csv(data: pd.DataFrame, path: Path) -> None:
    """Write one Phase 8 CSV in a consistent format."""

    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")


def validate_new_outputs(
    sample_overview: pd.DataFrame,
    regression_table: pd.DataFrame,
    sentiment_summary: pd.DataFrame,
    rq_summary: pd.DataFrame,
    limitations: pd.DataFrame,
    script_inventory: pd.DataFrame,
    manifest: pd.DataFrame,
    audit: pd.DataFrame,
) -> None:
    """Require final schemas and reconciliations before completion."""

    if len(sentiment_summary) != 5 or set(sentiment_summary["market"]) != set(
        MARKET_LABELS.values()
    ):
        raise RuntimeError("Final sentiment summary does not contain five markets.")
    if sentiment_summary.set_index("market")["N"].to_dict() != {
        MARKET_LABELS[key]: value for key, value in EXPECTED_REGRESSION_N.items()
    }:
        raise RuntimeError("Final sentiment summary sample sizes do not reconcile.")
    if not sentiment_summary["CI_includes_zero"].all():
        raise RuntimeError("Final sentiment summary inference does not reconcile.")
    if list(rq_summary["item"]) != ["RQ1", "RQ2", "H1"]:
        raise RuntimeError("Research-question summary schema changed.")
    if len(limitations) != 10:
        raise RuntimeError("The final limitations table must contain ten items.")
    if len(script_inventory) != 13:
        raise RuntimeError("The active script inventory must contain 13 scripts.")
    if regression_table["statistic"].eq("HAC maximum lag").sum() != 1:
        raise RuntimeError("Final regression table is missing the HAC metadata row.")
    if sample_overview.empty or sample_overview["value"].isna().any():
        raise RuntimeError("Final sample overview contains a missing value.")
    if not audit["status"].eq("PASS").all():
        raise RuntimeError("The final reproducibility audit contains a failure.")
    expected_phase8 = {
        relative(FINAL_SAMPLE_OVERVIEW_TABLE),
        relative(FINAL_REGRESSION_TABLE),
        relative(FINAL_SENTIMENT_RESULTS_SUMMARY_TABLE),
        relative(RESEARCH_QUESTION_SUMMARY_TABLE),
        relative(FINAL_METHODOLOGICAL_LIMITATIONS_TABLE),
        relative(ACTIVE_SCRIPT_INVENTORY_TABLE),
        relative(FINAL_REPRODUCIBILITY_AUDIT_FILE),
        relative(FINAL_RESULTS_REVIEW_FILE),
        relative(FINAL_THESIS_OUTPUT_MANIFEST_TABLE),
    }
    manifest_names = set(manifest["filename"])
    if not expected_phase8.issubset(manifest_names):
        raise RuntimeError("The final manifest omits a Phase 8 output.")


def main() -> None:
    """Build final reporting artifacts without rerunning any empirical model."""

    measured_hashes = validate_frozen_state()
    sources = read_sources()
    validate_sources(sources)

    sample_overview = build_sample_overview(sources)
    regression_table = build_final_regression_table(sources)
    sentiment_summary = build_sentiment_summary(sources)
    rq_summary = build_research_question_summary(sentiment_summary)
    limitations = build_limitations(sources)
    script_inventory = build_script_inventory()
    review_note = build_review_note(sources, sentiment_summary)

    tables = {
        FINAL_SAMPLE_OVERVIEW_TABLE: sample_overview,
        FINAL_REGRESSION_TABLE: regression_table,
        FINAL_SENTIMENT_RESULTS_SUMMARY_TABLE: sentiment_summary,
        RESEARCH_QUESTION_SUMMARY_TABLE: rq_summary,
        FINAL_METHODOLOGICAL_LIMITATIONS_TABLE: limitations,
        ACTIVE_SCRIPT_INVENTORY_TABLE: script_inventory,
    }
    for path, data in tables.items():
        write_csv(data, path)
    FINAL_RESULTS_REVIEW_FILE.write_text(review_note, encoding="utf-8", newline="\n")

    # Write an initial manifest so the repository audit can verify its path.
    manifest = build_manifest()
    write_csv(manifest, FINAL_THESIS_OUTPUT_MANIFEST_TABLE)
    audit = build_reproducibility_audit(measured_hashes, sources)
    write_csv(audit, FINAL_REPRODUCIBILITY_AUDIT_FILE)

    # Rebuild once so the audit itself is represented in the manifest.
    manifest = build_manifest()
    write_csv(manifest, FINAL_THESIS_OUTPUT_MANIFEST_TABLE)
    validate_new_outputs(
        sample_overview,
        regression_table,
        sentiment_summary,
        rq_summary,
        limitations,
        script_inventory,
        manifest,
        audit,
    )

    rechecked_hashes = validate_frozen_state()
    if rechecked_hashes != measured_hashes:
        raise RuntimeError("A frozen artifact changed while Phase 8 was running.")

    phase8_paths = [
        *tables,
        FINAL_THESIS_OUTPUT_MANIFEST_TABLE,
        FINAL_REPRODUCIBILITY_AUDIT_FILE,
        FINAL_RESULTS_REVIEW_FILE,
    ]
    if missing := [path for path in phase8_paths if not path.exists()]:
        raise RuntimeError(f"Phase 8 outputs are missing: {missing}")
    if empty := [path for path in phase8_paths if path.stat().st_size == 0]:
        raise RuntimeError(f"Phase 8 outputs are empty: {empty}")

    print("Frozen Phases 2-7 hashes verified.")
    print(f"Final manifest rows: {len(manifest)}")
    print(f"Reproducibility audit checks passed: {len(audit)}")
    print("Primary lagged-sentiment results:")
    print(
        sentiment_summary[
            ["market", "N", "sentiment_coefficient", "p_value", "CI_includes_zero"]
        ].to_string(index=False)
    )
    print("Phase 8 reporting consolidation completed without empirical re-estimation.")


if __name__ == "__main__":
    main()
