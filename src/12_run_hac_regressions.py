"""Estimate the five frozen Phase 7 OLS models with HAC inference."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels  # noqa: E402
import statsmodels.api as sm  # noqa: E402


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from config import (  # noqa: E402
    END_DATE,
    FINBERT_REDDIT_FILE,
    HAC_MAX_LAGS,
    MARKET_ALIGNED_LAGGED_FILE,
    MARKET_RETURNS_GARCH_FILE,
    MARKET_TICKERS,
    REGRESSION_COEFFICIENTS_LONG_TABLE,
    REGRESSION_MODEL_VALIDATION_FILE,
    REGRESSION_RESULTS_TABLE,
    REGRESSION_SAMPLE_DESCRIPTIVES_TABLE,
    REGRESSION_SAMPLE_YEAR_COMPOSITION_TABLE,
    SENTIMENT_COEFFICIENT_COMPARISON_FIGURE,
    SENTIMENT_COEFFICIENT_COMPARISON_TABLE,
    START_DATE,
)


EXPECTED_PHASE6_SHA256 = (
    "1071A9E9BD7E9322F843EE3A60E331B111C70BA386D49E59F2FCC0AC38EA6A86"
)
EXPECTED_PHASE5_SHA256 = (
    "FFCF4D30CE1A064B04E726B67273204CF53CDBF1EE5696F1F727251966C425DE"
)
EXPECTED_FINBERT_SHA256 = (
    "2E4A693558197B8007F81C5D348362140524E3C31A8D31293F86D691DDB9C7FF"
)
EXPECTED_MARKETS = ["EURO_STOXX_50", "DAX", "CAC_40", "FTSE_100", "WIG20"]
EXPECTED_OBSERVATIONS = {
    "EURO_STOXX_50": 401,
    "DAX": 406,
    "CAC_40": 407,
    "FTSE_100": 397,
    "WIG20": 402,
}
PANEL_COLUMNS = [
    "date",
    "index_name",
    "ticker",
    "data_source",
    "close_level",
    "log_return",
    "return_pct",
    "garch_volatility",
    "sentiment",
    "attention",
    "sentiment_lag1",
    "attention_lag1",
    "volatility_lag1",
    "return_lag1",
    "regression_eligible",
]
DEPENDENT_VARIABLE = "garch_volatility"
REGRESSORS = [
    "sentiment_lag1",
    "attention_lag1",
    "volatility_lag1",
    "return_lag1",
]
MODEL_VARIABLES = [DEPENDENT_VARIABLE, *REGRESSORS]
TERMS = ["const", *REGRESSORS]
LAG_SOURCE_COLUMNS = {
    "sentiment_lag1": "sentiment",
    "attention_lag1": "attention",
    "volatility_lag1": "garch_volatility",
    "return_lag1": "log_return",
}
DESCRIPTIVE_VARIABLES = MODEL_VARIABLES
FIGURE_DPI = 300
NUMERIC_TOLERANCE = 1e-12


def file_sha256(path: Path) -> str:
    """Return an uppercase SHA-256 digest for one file."""

    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def finite_values(data: pd.DataFrame | pd.Series) -> bool:
    """Return whether every supplied numeric value is finite."""

    numeric = data.apply(pd.to_numeric, errors="coerce")
    return bool(np.isfinite(numeric.to_numpy(dtype=float)).all())


def numeric_series_equal(left: pd.Series, right: pd.Series) -> bool:
    """Compare numeric series within CSV-safe floating precision."""

    return bool(
        np.allclose(
            pd.to_numeric(left, errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(right, errors="coerce").to_numpy(dtype=float),
            rtol=0,
            atol=NUMERIC_TOLERANCE,
            equal_nan=True,
        )
    )


def parse_boolean(series: pd.Series, column_name: str) -> pd.Series:
    """Parse a CSV Boolean column without treating nonempty strings as true."""

    if series.dtype == bool:
        return series.copy()
    normalized = series.astype("string").str.strip().str.lower()
    if normalized.isna().any() or not normalized.isin(["true", "false"]).all():
        raise ValueError(f"{column_name} contains an invalid Boolean value.")
    return normalized.eq("true")


def validate_configuration() -> None:
    """Require exactly the frozen Phase 7 specification and output locations."""

    if START_DATE != "2021-01-01" or END_DATE != "2023-12-31":
        raise ValueError("Phase 7 must cover 2021-01-01 through 2023-12-31.")
    if list(MARKET_TICKERS) != EXPECTED_MARKETS:
        raise ValueError("The approved market order or membership changed.")
    if HAC_MAX_LAGS != 5:
        raise ValueError("Phase 7 requires HAC/Newey-West maximum lag 5 exactly.")
    output_paths = [
        REGRESSION_COEFFICIENTS_LONG_TABLE,
        REGRESSION_RESULTS_TABLE,
        SENTIMENT_COEFFICIENT_COMPARISON_TABLE,
        REGRESSION_SAMPLE_DESCRIPTIVES_TABLE,
        REGRESSION_SAMPLE_YEAR_COMPOSITION_TABLE,
        REGRESSION_MODEL_VALIDATION_FILE,
        SENTIMENT_COEFFICIENT_COMPARISON_FIGURE,
    ]
    if len(output_paths) != len(set(output_paths)):
        raise ValueError("Phase 7 output paths must be unique.")


def validate_frozen_hashes() -> dict[str, str]:
    """Verify every frozen empirical input required by the Phase 7 checkpoint."""

    paths = {
        "phase6_aligned_lagged": MARKET_ALIGNED_LAGGED_FILE,
        "phase5_market_garch": MARKET_RETURNS_GARCH_FILE,
        "finbert_post_level": FINBERT_REDDIT_FILE,
    }
    expected = {
        "phase6_aligned_lagged": EXPECTED_PHASE6_SHA256,
        "phase5_market_garch": EXPECTED_PHASE5_SHA256,
        "finbert_post_level": EXPECTED_FINBERT_SHA256,
    }
    hashes: dict[str, str] = {}
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Required frozen input is missing: {path}")
        hashes[name] = file_sha256(path)
        if hashes[name] != expected[name]:
            raise ValueError(
                f"Frozen {name} SHA-256 changed. Expected {expected[name]}, "
                f"found {hashes[name]}."
            )
    return hashes


def load_and_validate_samples() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Load the frozen Phase 6 panel and reconstruct eligible market samples."""

    panel = pd.read_csv(MARKET_ALIGNED_LAGGED_FILE, parse_dates=["date"])
    if list(panel.columns) != PANEL_COLUMNS:
        raise ValueError("The frozen Phase 6 panel schema changed.")
    panel["regression_eligible"] = parse_boolean(
        panel["regression_eligible"], "regression_eligible"
    )
    if set(panel["index_name"]) != set(EXPECTED_MARKETS):
        raise ValueError("The Phase 6 panel does not contain exactly five markets.")
    if panel.duplicated(["index_name", "date"]).any():
        raise ValueError("The Phase 6 panel contains duplicate market/date rows.")
    if panel["date"].isna().any() or not panel["date"].between(
        START_DATE, END_DATE, inclusive="both"
    ).all():
        raise ValueError("The Phase 6 panel contains invalid or out-of-period dates.")

    samples: dict[str, pd.DataFrame] = {}
    for index_name in EXPECTED_MARKETS:
        full_market = panel.loc[panel["index_name"].eq(index_name)].copy()
        full_market = full_market.sort_values("date", kind="stable").reset_index(
            drop=True
        )
        if not full_market["date"].is_unique or not full_market[
            "date"
        ].is_monotonic_increasing:
            raise ValueError(f"{index_name} trading dates are not unique and sorted.")

        for lag_column, source_column in LAG_SOURCE_COLUMNS.items():
            if not numeric_series_equal(
                full_market[lag_column], full_market[source_column].shift(1)
            ):
                raise ValueError(
                    f"{index_name} {lag_column} does not equal the prior trading "
                    f"row's {source_column}."
                )

        recomputed_eligible = pd.Series(True, index=full_market.index)
        for variable in MODEL_VARIABLES:
            numeric = pd.to_numeric(full_market[variable], errors="coerce")
            recomputed_eligible &= np.isfinite(numeric.to_numpy(dtype=float))
        if not full_market["regression_eligible"].equals(recomputed_eligible):
            raise ValueError(f"{index_name} regression eligibility no longer reproduces.")

        sample = full_market.loc[full_market["regression_eligible"]].copy()
        expected_n = EXPECTED_OBSERVATIONS[index_name]
        if len(sample) != expected_n:
            raise ValueError(
                f"{index_name} expected {expected_n} eligible rows, found {len(sample)}."
            )
        if sample[MODEL_VARIABLES].isna().any().any():
            raise ValueError(f"{index_name} has missing regression input values.")
        if not finite_values(sample[MODEL_VARIABLES]):
            raise ValueError(f"{index_name} has infinite regression input values.")
        if not sample["attention_lag1"].gt(0).all():
            raise ValueError(f"{index_name} has nonpositive eligible attention_lag1.")
        if not sample["garch_volatility"].gt(0).all():
            raise ValueError(f"{index_name} has nonpositive eligible volatility.")
        if not sample["sentiment_lag1"].between(-1, 1, inclusive="both").all():
            raise ValueError(f"{index_name} has out-of-range lagged sentiment.")
        if sample["index_name"].nunique() != 1 or sample[
            "index_name"
        ].iloc[0] != index_name:
            raise ValueError(f"{index_name} sample contains another market.")
        samples[index_name] = sample.reset_index(drop=True)
    return panel, samples


def estimate_models(
    panel: pd.DataFrame,
    samples: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate exactly one approved OLS-HAC model for each market."""

    coefficient_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    for index_name in EXPECTED_MARKETS:
        sample = samples[index_name]
        full_market = panel.loc[panel["index_name"].eq(index_name)].copy()
        full_market = full_market.sort_values("date", kind="stable").reset_index(
            drop=True
        )
        return_lag_is_decimal = numeric_series_equal(
            full_market["return_lag1"], full_market["log_return"].shift(1)
        )
        all_lags_match_prior_trading_row = all(
            numeric_series_equal(
                full_market[lag_column], full_market[source_column].shift(1)
            )
            for lag_column, source_column in LAG_SOURCE_COLUMNS.items()
        )
        y = sample[DEPENDENT_VARIABLE].astype(float)
        predictors = sample[REGRESSORS].astype(float)
        X = sm.add_constant(predictors, has_constant="add")
        X = X.loc[:, TERMS]
        matrix_rank = int(np.linalg.matrix_rank(X.to_numpy(dtype=float)))
        if matrix_rank != len(TERMS):
            raise ValueError(f"{index_name} model matrix is rank deficient.")

        result = sm.OLS(y, X, missing="raise").fit(
            cov_type="HAC",
            cov_kwds={"maxlags": HAC_MAX_LAGS},
        )
        if int(result.nobs) != EXPECTED_OBSERVATIONS[index_name]:
            raise RuntimeError(f"{index_name} fitted model silently changed N.")
        if list(result.params.index) != TERMS:
            raise RuntimeError(f"{index_name} fitted terms differ from the specification.")

        confidence = result.conf_int(alpha=0.05)
        for term in TERMS:
            coefficient_rows.append(
                {
                    "index_name": index_name,
                    "term": term,
                    "coefficient": float(result.params[term]),
                    "hac_standard_error": float(result.bse[term]),
                    "t_statistic": float(result.tvalues[term]),
                    "p_value": float(result.pvalues[term]),
                    "ci_lower_95": float(confidence.loc[term, 0]),
                    "ci_upper_95": float(confidence.loc[term, 1]),
                    "nobs": int(result.nobs),
                    "r_squared": float(result.rsquared),
                    "hac_max_lags": HAC_MAX_LAGS,
                }
            )

        coefficient_values = result.params.to_numpy(dtype=float)
        standard_errors = result.bse.to_numpy(dtype=float)
        p_values = result.pvalues.to_numpy(dtype=float)
        confidence_values = confidence.to_numpy(dtype=float)
        covariance_type = str(result.cov_type)
        fitted_hac_lags = int(result.cov_kwds.get("maxlags", -1))
        checks = {
            "sample_dates_ordered": bool(sample["date"].is_monotonic_increasing),
            "sample_contains_only_own_market": bool(
                sample["index_name"].eq(index_name).all()
            ),
            "no_nan_input": bool(not sample[MODEL_VARIABLES].isna().any().any()),
            "no_infinite_input": finite_values(sample[MODEL_VARIABLES]),
            "attention_lag1_positive": bool(sample["attention_lag1"].gt(0).all()),
            "intercept_present": "const" in X.columns,
            "exact_regressor_list": list(X.columns) == TERMS,
            "hac_covariance_active": covariance_type.upper() == "HAC",
            "hac_max_lag_exact": fitted_hac_lags == HAC_MAX_LAGS == 5,
            "no_rows_silently_dropped": int(result.nobs) == len(sample),
            "coefficients_finite": bool(np.isfinite(coefficient_values).all()),
            "standard_errors_finite_nonnegative": bool(
                np.isfinite(standard_errors).all() and (standard_errors >= 0).all()
            ),
            "p_values_valid": bool(
                np.isfinite(p_values).all()
                and (p_values >= 0).all()
                and (p_values <= 1).all()
            ),
            "confidence_intervals_valid": bool(
                np.isfinite(confidence_values).all()
                and (confidence_values[:, 0] <= confidence_values[:, 1]).all()
            ),
            "r_squared_finite": bool(np.isfinite(result.rsquared)),
            "model_matrix_full_rank": matrix_rank == X.shape[1],
            "return_lag_is_decimal_log_return": return_lag_is_decimal,
            "lagged_predictors_match_prior_trading_row": (
                all_lags_match_prior_trading_row
            ),
            "lagged_predictors_only": list(X.columns) == TERMS,
        }
        validation_rows.append(
            {
                "index_name": index_name,
                "expected_n": EXPECTED_OBSERVATIONS[index_name],
                "fitted_model_nobs": int(result.nobs),
                "first_model_date": sample["date"].min(),
                "last_model_date": sample["date"].max(),
                "dependent_variable": DEPENDENT_VARIABLE,
                "regressor_list": "|".join(REGRESSORS),
                "intercept_present": checks["intercept_present"],
                "statsmodels_version": statsmodels.__version__,
                "covariance_type": covariance_type,
                "hac_max_lags": fitted_hac_lags,
                "model_matrix_rank": matrix_rank,
                "model_matrix_columns": X.shape[1],
                **checks,
                "validation_status": (
                    "PASS" if all(bool(value) for value in checks.values()) else "FAIL"
                ),
            }
        )

    coefficients = pd.DataFrame(coefficient_rows)
    validation = pd.DataFrame(validation_rows)
    if not validation["validation_status"].eq("PASS").all():
        failed = validation.loc[
            validation["validation_status"].ne("PASS"), "index_name"
        ].tolist()
        raise RuntimeError(f"Phase 7 model validation failed for: {failed}")
    return coefficients, validation


def build_regression_results(coefficients: pd.DataFrame) -> pd.DataFrame:
    """Build one thesis-ready wide table with exact numeric inference."""

    rows: list[dict[str, Any]] = []
    statistics = [
        ("coefficient", "coefficient"),
        ("HAC standard error", "hac_standard_error"),
        ("p-value", "p_value"),
        ("95% CI lower", "ci_lower_95"),
        ("95% CI upper", "ci_upper_95"),
    ]
    for term in TERMS:
        subset = coefficients.loc[coefficients["term"].eq(term)].set_index(
            "index_name"
        )
        for statistic_label, source_column in statistics:
            row: dict[str, Any] = {"term": term, "statistic": statistic_label}
            row.update(
                {
                    index_name: float(subset.loc[index_name, source_column])
                    for index_name in EXPECTED_MARKETS
                }
            )
            rows.append(row)

    model_rows = [
        ("observations", "nobs"),
        ("R-squared", "r_squared"),
        ("HAC max lag", "hac_max_lags"),
    ]
    for statistic_label, source_column in model_rows:
        row = {"term": "model", "statistic": statistic_label}
        row.update(
            {
                index_name: float(
                    coefficients.loc[
                        coefficients["index_name"].eq(index_name), source_column
                    ].iloc[0]
                )
                for index_name in EXPECTED_MARKETS
            }
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=["term", "statistic", *EXPECTED_MARKETS])


def build_sentiment_comparison(coefficients: pd.DataFrame) -> pd.DataFrame:
    """Extract the primary coefficient and transparent sign/interval metadata."""

    comparison = coefficients.loc[
        coefficients["term"].eq("sentiment_lag1"),
        [
            "index_name",
            "coefficient",
            "hac_standard_error",
            "p_value",
            "ci_lower_95",
            "ci_upper_95",
            "nobs",
        ],
    ].copy()
    comparison["expected_sign_negative"] = True
    comparison["observed_sign"] = np.select(
        [comparison["coefficient"].lt(0), comparison["coefficient"].gt(0)],
        ["negative", "positive"],
        default="zero",
    )
    comparison["ci_excludes_zero"] = (
        comparison["ci_lower_95"].gt(0) | comparison["ci_upper_95"].lt(0)
    )
    order = pd.Categorical(
        comparison["index_name"], categories=EXPECTED_MARKETS, ordered=True
    )
    return comparison.assign(_order=order).sort_values("_order").drop(
        columns="_order"
    )


def build_sample_descriptives(
    samples: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Describe the variables in the actual market-specific estimation samples."""

    rows: list[dict[str, Any]] = []
    for index_name in EXPECTED_MARKETS:
        sample = samples[index_name]
        for variable in DESCRIPTIVE_VARIABLES:
            values = sample[variable].astype(float)
            rows.append(
                {
                    "index_name": index_name,
                    "variable": variable,
                    "n": int(values.count()),
                    "mean": float(values.mean()),
                    "std": float(values.std()),
                    "min": float(values.min()),
                    "25%": float(values.quantile(0.25)),
                    "median": float(values.median()),
                    "75%": float(values.quantile(0.75)),
                    "max": float(values.max()),
                }
            )
    return pd.DataFrame(rows)


def build_year_composition(samples: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Report the unweighted year composition of each actual regression sample."""

    rows: list[dict[str, Any]] = []
    for index_name in EXPECTED_MARKETS:
        sample = samples[index_name]
        years = sample["date"].dt.year.value_counts()
        for year in [2021, 2022, 2023]:
            n = int(years.get(year, 0))
            rows.append(
                {
                    "index_name": index_name,
                    "year": year,
                    "n": n,
                    "share_of_market_regression_sample": float(n / len(sample)),
                }
            )
    return pd.DataFrame(rows)


def configure_plots() -> None:
    """Apply the repository's restrained thesis plotting style."""

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
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


def plot_sentiment_comparison(comparison: pd.DataFrame) -> None:
    """Create the approved cross-market forest plot for lagged sentiment."""

    ordered = comparison.set_index("index_name").loc[EXPECTED_MARKETS].reset_index()
    estimates = ordered["coefficient"].to_numpy(dtype=float)
    lower = ordered["ci_lower_95"].to_numpy(dtype=float)
    upper = ordered["ci_upper_95"].to_numpy(dtype=float)
    positions = np.arange(len(ordered))[::-1]
    errors = np.vstack([estimates - lower, upper - estimates])

    configure_plots()
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.errorbar(
        estimates,
        positions,
        xerr=errors,
        fmt="o",
        color="#1D4ED8",
        ecolor="#1D4ED8",
        elinewidth=1.4,
        capsize=4,
        markersize=6,
    )
    ax.axvline(0, color="#111827", linewidth=1, linestyle="--")
    ax.set_yticks(positions)
    ax.set_yticklabels([name.replace("_", " ") for name in EXPECTED_MARKETS])
    ax.set_xlabel("Coefficient on lagged sentiment (sentiment_lag1)")
    ax.set_title("Lagged Reddit sentiment coefficients across European markets")
    axis_extent = max(abs(float(lower.min())), abs(float(upper.max())), 0.01) * 1.12
    ax.set_xlim(-axis_extent, axis_extent)
    ax.grid(axis="x", visible=True)
    ax.grid(axis="y", visible=False)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.text(
        0.5,
        0.015,
        "Point estimates from five separate OLS models; bars are 95% HAC/Newey-West confidence intervals (max lag 5).",
        ha="center",
        fontsize=8.5,
        color="#374151",
    )
    fig.tight_layout(rect=[0, 0.045, 1, 1])
    SENTIMENT_COEFFICIENT_COMPARISON_FIGURE.parent.mkdir(
        parents=True, exist_ok=True
    )
    fig.savefig(
        SENTIMENT_COEFFICIENT_COMPARISON_FIGURE,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )
    plt.close(fig)


def write_csv(data: pd.DataFrame, path: Path) -> None:
    """Write a validated Phase 7 CSV with round-trip-safe numeric precision."""

    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(
        path,
        index=False,
        date_format="%Y-%m-%d",
        float_format="%.17g",
        na_rep="",
    )


def validate_output_tables(
    coefficients: pd.DataFrame,
    regression_results: pd.DataFrame,
    comparison: pd.DataFrame,
    descriptives: pd.DataFrame,
    year_composition: pd.DataFrame,
    validation: pd.DataFrame,
) -> None:
    """Require complete schemas and internally reconciled Phase 7 outputs."""

    if len(coefficients) != len(EXPECTED_MARKETS) * len(TERMS):
        raise RuntimeError("The long coefficient table does not have 25 rows.")
    if coefficients.duplicated(["index_name", "term"]).any():
        raise RuntimeError("The long coefficient table has duplicate market/terms.")
    if set(coefficients["term"]) != set(TERMS):
        raise RuntimeError("The long coefficient table is missing an approved term.")
    if len(comparison) != len(EXPECTED_MARKETS):
        raise RuntimeError("The sentiment comparison does not have five markets.")
    if len(descriptives) != len(EXPECTED_MARKETS) * len(DESCRIPTIVE_VARIABLES):
        raise RuntimeError("The regression descriptives do not have 25 rows.")
    if len(year_composition) != len(EXPECTED_MARKETS) * 3:
        raise RuntimeError("The year composition does not have 15 rows.")
    shares = year_composition.groupby("index_name", sort=False)[
        "share_of_market_regression_sample"
    ].sum()
    if not np.allclose(shares, 1.0, rtol=0, atol=NUMERIC_TOLERANCE):
        raise RuntimeError("A market's regression-sample year shares do not sum to 1.")
    if not validation["validation_status"].eq("PASS").all():
        raise RuntimeError("The regression model diagnostic contains a failure.")
    expected_result_rows = len(TERMS) * 5 + 3
    if len(regression_results) != expected_result_rows:
        raise RuntimeError("The thesis-ready regression table has an invalid shape.")


def main() -> None:
    """Run only the five frozen Phase 7 regressions and their reporting outputs."""

    validate_configuration()
    frozen_hashes = validate_frozen_hashes()
    panel, samples = load_and_validate_samples()
    coefficients, validation = estimate_models(panel, samples)
    regression_results = build_regression_results(coefficients)
    comparison = build_sentiment_comparison(coefficients)
    descriptives = build_sample_descriptives(samples)
    year_composition = build_year_composition(samples)
    validate_output_tables(
        coefficients,
        regression_results,
        comparison,
        descriptives,
        year_composition,
        validation,
    )

    outputs = {
        REGRESSION_COEFFICIENTS_LONG_TABLE: coefficients,
        REGRESSION_RESULTS_TABLE: regression_results,
        SENTIMENT_COEFFICIENT_COMPARISON_TABLE: comparison,
        REGRESSION_SAMPLE_DESCRIPTIVES_TABLE: descriptives,
        REGRESSION_SAMPLE_YEAR_COMPOSITION_TABLE: year_composition,
        REGRESSION_MODEL_VALIDATION_FILE: validation,
    }
    for path, data in outputs.items():
        write_csv(data, path)
    plot_sentiment_comparison(comparison)

    rechecked_hashes = validate_frozen_hashes()
    if rechecked_hashes != frozen_hashes:
        raise RuntimeError("A frozen empirical input changed during Phase 7.")
    required_paths = [*outputs, SENTIMENT_COEFFICIENT_COMPARISON_FIGURE]
    if missing := [path for path in required_paths if not path.exists()]:
        raise RuntimeError(f"Phase 7 outputs are missing: {missing}")
    if empty := [path for path in required_paths if path.stat().st_size == 0]:
        raise RuntimeError(f"Phase 7 outputs are empty: {empty}")

    print(f"statsmodels version: {statsmodels.__version__}")
    print(f"Validated frozen Phase 6 SHA-256: {frozen_hashes['phase6_aligned_lagged']}")
    print("\nPrimary lagged-sentiment coefficients:")
    print(
        comparison[
            [
                "index_name",
                "coefficient",
                "hac_standard_error",
                "p_value",
                "ci_lower_95",
                "ci_upper_95",
                "nobs",
            ]
        ].to_string(index=False)
    )
    print(f"\nSaved long coefficient results to {REGRESSION_COEFFICIENTS_LONG_TABLE}")
    print(
        "Coefficient results SHA-256: "
        f"{file_sha256(REGRESSION_COEFFICIENTS_LONG_TABLE)}"
    )
    print("Phase 7 five separate OLS-HAC regressions completed successfully.")
    print("Phase 7 computation complete; no Phase 8 work was performed.")


if __name__ == "__main__":
    main()
