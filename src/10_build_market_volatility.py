"""Build Phase 5 market returns and GARCH(1,1)-Student-t volatility."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import arch  # noqa: E402
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from arch import arch_model  # noqa: E402


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from config import (  # noqa: E402
    END_DATE,
    GARCH_CONDITIONAL_VOLATILITY_FIGURE,
    GARCH_DISTRIBUTION,
    GARCH_MEAN_MODEL,
    GARCH_MODEL_DIAGNOSTICS_FILE,
    GARCH_O,
    GARCH_P,
    GARCH_PARAMETERS_TABLE,
    GARCH_Q,
    GARCH_VOLATILITY_DESCRIPTIVES_TABLE,
    GARCH_VOLATILITY_MODEL,
    MARKET_DATA_SOURCES,
    MARKET_LOG_RETURNS_FIGURE,
    MARKET_PRICES_FILE,
    MARKET_RETURN_DESCRIPTIVES_TABLE,
    MARKET_RETURNS_GARCH_FILE,
    MARKET_TICKERS,
    START_DATE,
)


EXPECTED_INPUT_SHA256 = (
    "37342707BBB8E19FBDEF363B16C0372ED4420B22A2D7F56829BE166B162443AC"
)
EXPECTED_MARKETS = ["EURO_STOXX_50", "DAX", "CAC_40", "FTSE_100", "WIG20"]
EXPECTED_PRICE_COUNTS = {
    "EURO_STOXX_50": 759,
    "DAX": 767,
    "CAC_40": 770,
    "FTSE_100": 754,
    "WIG20": 752,
}
EXPECTED_RETURN_COUNTS = {
    market: count - 1 for market, count in EXPECTED_PRICE_COUNTS.items()
}
INPUT_COLUMNS = ["date", "index_name", "ticker", "data_source", "close_level"]
OUTPUT_COLUMNS = [
    *INPUT_COLUMNS,
    "log_return",
    "return_pct",
    "garch_volatility",
]
PARAMETER_NAMES = ["mu", "omega", "alpha[1]", "beta[1]", "nu"]
MODEL_SPECIFICATION = "Constant mean; GARCH(1,1); Student-t innovations"
RETURN_IDENTITY_TOLERANCE = 1e-12
FIGURE_DPI = 300


def file_sha256(path: Path) -> str:
    """Return an uppercase SHA-256 digest for one file."""

    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def validate_configuration() -> None:
    """Require exactly the frozen Phase 5 market and GARCH specification."""

    if START_DATE != "2021-01-01" or END_DATE != "2023-12-31":
        raise ValueError("Phase 5 must cover 2021-01-01 through 2023-12-31.")
    if list(MARKET_TICKERS) != EXPECTED_MARKETS:
        raise ValueError("The configured market order or membership changed.")
    if set(MARKET_DATA_SOURCES) != set(EXPECTED_MARKETS):
        raise ValueError("Market source configuration is incomplete.")
    if MARKET_TICKERS.get("WIG20") != "WIG20":
        raise ValueError("WIG20 must use the stored ticker WIG20.")
    if MARKET_DATA_SOURCES.get("WIG20") != "Stooq":
        raise ValueError("WIG20 must retain Stooq provenance.")
    specification = (
        GARCH_MEAN_MODEL,
        GARCH_VOLATILITY_MODEL,
        GARCH_P,
        GARCH_O,
        GARCH_Q,
        GARCH_DISTRIBUTION,
    )
    expected = ("Constant", "GARCH", 1, 0, 1, "StudentsT")
    if specification != expected:
        raise ValueError(
            "Phase 5 requires constant-mean GARCH(1,1) with Student-t innovations."
        )


def load_and_validate_prices() -> tuple[pd.DataFrame, str]:
    """Load the immutable market-price panel and validate its identity."""

    if not MARKET_PRICES_FILE.exists():
        raise FileNotFoundError(f"Market-price input is missing: {MARKET_PRICES_FILE}")
    input_sha256 = file_sha256(MARKET_PRICES_FILE)
    if input_sha256 != EXPECTED_INPUT_SHA256:
        raise ValueError(
            "Immutable market-price input changed. Expected "
            f"{EXPECTED_INPUT_SHA256}, found {input_sha256}."
        )

    data = pd.read_csv(MARKET_PRICES_FILE, parse_dates=["date"])
    if list(data.columns) != INPUT_COLUMNS:
        raise ValueError("Market-price input schema changed.")
    if data.empty or set(data["index_name"].unique()) != set(EXPECTED_MARKETS):
        raise ValueError("Market-price input does not contain the five approved markets.")
    if data.duplicated(["index_name", "date"]).any():
        raise ValueError("Market-price input contains duplicate market/date rows.")
    if data["date"].isna().any() or not data["date"].between(
        START_DATE, END_DATE, inclusive="both"
    ).all():
        raise ValueError("Market-price input contains an invalid sample date.")

    close_level = pd.to_numeric(data["close_level"], errors="coerce")
    if close_level.isna().any() or not np.isfinite(close_level.to_numpy()).all():
        raise ValueError("Market close levels contain missing or non-finite values.")
    if close_level.le(0).any():
        raise ValueError("Market close levels must be strictly positive.")
    data = data.copy()
    data["close_level"] = close_level

    prohibited = data.astype(str).apply(
        lambda column: column.str.contains(
            "WIG20.WA|GPW.WA", case=False, regex=True
        )
    )
    if prohibited.any().any():
        raise ValueError("A prohibited WIG20 proxy occurs in market-price input.")

    ordered_groups: list[pd.DataFrame] = []
    for index_name in EXPECTED_MARKETS:
        subset = data.loc[data["index_name"].eq(index_name)].copy()
        subset = subset.sort_values("date", kind="stable")
        if len(subset) != EXPECTED_PRICE_COUNTS[index_name]:
            raise ValueError(
                f"{index_name} has {len(subset)} prices; expected "
                f"{EXPECTED_PRICE_COUNTS[index_name]}."
            )
        if not subset["date"].is_monotonic_increasing or not subset[
            "date"
        ].is_unique:
            raise ValueError(f"{index_name} dates are not strictly ordered.")
        if set(subset["ticker"].astype(str)) != {MARKET_TICKERS[index_name]}:
            raise ValueError(f"{index_name} has an unexpected ticker.")
        if set(subset["data_source"].astype(str)) != {
            MARKET_DATA_SOURCES[index_name]
        }:
            raise ValueError(f"{index_name} has an unexpected data source.")
        if subset["date"].min() != pd.Timestamp("2021-01-04"):
            raise ValueError(f"{index_name} has an unexpected first trading date.")
        if subset["date"].max() != pd.Timestamp("2023-12-29"):
            raise ValueError(f"{index_name} has an unexpected last trading date.")
        ordered_groups.append(subset)
    return pd.concat(ordered_groups, ignore_index=True), input_sha256


def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate unmodified log returns strictly within each market."""

    groups: list[pd.DataFrame] = []
    for index_name in EXPECTED_MARKETS:
        subset = prices.loc[prices["index_name"].eq(index_name)].copy()
        subset = subset.sort_values("date", kind="stable").reset_index(drop=True)
        previous_close = subset["close_level"].shift(1)
        subset["log_return"] = np.log(subset["close_level"] / previous_close)
        subset["return_pct"] = 100.0 * subset["log_return"]
        subset["garch_volatility"] = np.nan
        groups.append(subset)
    returns = pd.concat(groups, ignore_index=True)
    validate_returns(returns)
    return returns


def validate_returns(data: pd.DataFrame) -> None:
    """Reconcile within-market return counts, missingness, and identities."""

    if list(data.columns) != OUTPUT_COLUMNS:
        raise ValueError("The Phase 5 return panel has an invalid schema.")
    if data.duplicated(["index_name", "date"]).any():
        raise ValueError("The return panel contains duplicate market/date rows.")
    for index_name in EXPECTED_MARKETS:
        subset = data.loc[data["index_name"].eq(index_name)].copy()
        subset = subset.sort_values("date", kind="stable").reset_index(drop=True)
        missing_return = subset["log_return"].isna()
        missing_pct = subset["return_pct"].isna()
        if int(missing_return.sum()) != 1 or not bool(missing_return.iloc[0]):
            raise ValueError(
                f"{index_name} must have only its first log return missing."
            )
        if not missing_return.equals(missing_pct):
            raise ValueError(f"{index_name} return missingness is inconsistent.")
        finite_log_return = subset.loc[~missing_return, "log_return"].to_numpy()
        finite_return_pct = subset.loc[~missing_pct, "return_pct"].to_numpy()
        if len(finite_log_return) != EXPECTED_RETURN_COUNTS[index_name]:
            raise ValueError(f"{index_name} finite return count does not reconcile.")
        if not np.isfinite(finite_log_return).all() or not np.isfinite(
            finite_return_pct
        ).all():
            raise ValueError(f"{index_name} contains a non-finite return.")
        expected = np.log(
            subset["close_level"].to_numpy()[1:]
            / subset["close_level"].to_numpy()[:-1]
        )
        if not np.allclose(
            finite_log_return,
            expected,
            atol=RETURN_IDENTITY_TOLERANCE,
            rtol=0,
        ):
            raise ValueError(f"{index_name} log returns do not reproduce price ratios.")
        if not np.allclose(
            finite_return_pct,
            100.0 * finite_log_return,
            atol=RETURN_IDENTITY_TOLERANCE,
            rtol=0,
        ):
            raise ValueError(f"{index_name} percentage returns are inconsistent.")


def optimizer_value(result: Any, name: str, default: Any = "") -> Any:
    """Read one optimizer field exposed by SciPy's OptimizeResult."""

    optimization_result = result.optimization_result
    if hasattr(optimization_result, name):
        return getattr(optimization_result, name)
    return optimization_result.get(name, default)


def estimate_garch_models(
    return_panel: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Estimate exactly one approved GARCH model per market."""

    output = return_panel.copy()
    parameter_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []

    for index_name in EXPECTED_MARKETS:
        market_rows = output.index[output["index_name"].eq(index_name)]
        estimation_rows = market_rows[output.loc[market_rows, "return_pct"].notna()]
        dates = pd.DatetimeIndex(output.loc[estimation_rows, "date"])
        return_pct = pd.Series(
            output.loc[estimation_rows, "return_pct"].to_numpy(dtype=float),
            index=dates,
            name="return_pct",
        )
        model = arch_model(
            return_pct,
            mean=GARCH_MEAN_MODEL,
            vol=GARCH_VOLATILITY_MODEL,
            p=GARCH_P,
            o=GARCH_O,
            q=GARCH_Q,
            dist=GARCH_DISTRIBUTION,
            rescale=False,
        )
        result = model.fit(disp="off")
        missing_parameters = set(PARAMETER_NAMES) - set(result.params.index)
        if missing_parameters:
            raise ValueError(
                f"{index_name} GARCH result lacks parameters: "
                f"{sorted(missing_parameters)}."
            )
        parameters = result.params.reindex(PARAMETER_NAMES).astype(float)
        parameters_all_finite = bool(np.isfinite(parameters.to_numpy()).all())
        convergence_flag = int(result.convergence_flag)
        optimizer_success = bool(optimizer_value(result, "success", False))
        optimizer_status = int(
            optimizer_value(result, "status", convergence_flag)
        )
        optimizer_message = str(optimizer_value(result, "message", ""))
        converged = convergence_flag == 0 and optimizer_success
        conditional_volatility = result.conditional_volatility
        if not isinstance(conditional_volatility, pd.Series):
            conditional_volatility = pd.Series(
                conditional_volatility, index=return_pct.index
            )
        volatility_values = conditional_volatility.to_numpy(dtype=float)
        volatility_positive_finite = bool(
            np.isfinite(volatility_values).all()
            and (volatility_values > 0).all()
        )
        dates_align = bool(conditional_volatility.index.equals(return_pct.index))
        nobs_matches = int(result.nobs) == len(return_pct)

        if not converged:
            raise RuntimeError(
                f"{index_name} GARCH failed to converge: flag={convergence_flag}, "
                f"success={optimizer_success}, status={optimizer_status}, "
                f"message={optimizer_message}"
            )
        if not parameters_all_finite:
            raise RuntimeError(f"{index_name} GARCH returned a non-finite parameter.")
        if not nobs_matches or len(volatility_values) != len(return_pct):
            raise RuntimeError(f"{index_name} GARCH sample length does not reconcile.")
        if not dates_align:
            raise RuntimeError(f"{index_name} volatility dates do not align to returns.")
        if not volatility_positive_finite:
            raise RuntimeError(
                f"{index_name} conditional volatility is not positive and finite."
            )

        output.loc[estimation_rows, "garch_volatility"] = volatility_values
        persistence = float(parameters["alpha[1]"] + parameters["beta[1]"])
        maximum_position = int(np.argmax(volatility_values))
        parameter_rows.append(
            {
                "index_name": index_name,
                "ticker": output.loc[market_rows[0], "ticker"],
                "estimation_input_units": "percentage returns",
                "model_specification": MODEL_SPECIFICATION,
                "mu": float(parameters["mu"]),
                "omega": float(parameters["omega"]),
                "alpha": float(parameters["alpha[1]"]),
                "beta": float(parameters["beta[1]"]),
                "alpha_plus_beta": persistence,
                "nu": float(parameters["nu"]),
                "nobs": int(result.nobs),
            }
        )
        diagnostic_rows.append(
            {
                "index_name": index_name,
                "ticker": output.loc[market_rows[0], "ticker"],
                "model_specification": MODEL_SPECIFICATION,
                "estimation_input_units": "percentage returns",
                "arch_version": arch.__version__,
                "observations": int(result.nobs),
                "estimation_start_date": dates.min(),
                "estimation_end_date": dates.max(),
                "converged": converged,
                "convergence_flag": convergence_flag,
                "optimizer_status": optimizer_status,
                "optimizer_success": optimizer_success,
                "optimizer_message": optimizer_message,
                "optimizer_iterations": optimizer_value(result, "nit", np.nan),
                "parameters_all_finite": parameters_all_finite,
                "conditional_volatility_positive_finite": (
                    volatility_positive_finite
                ),
                "conditional_volatility_dates_align": dates_align,
                "minimum_conditional_volatility": float(volatility_values.min()),
                "maximum_conditional_volatility": float(volatility_values.max()),
                "maximum_conditional_volatility_date": dates[maximum_position],
                "log_likelihood": float(result.loglikelihood),
                "aic": float(result.aic),
                "bic": float(result.bic),
            }
        )

    parameters = pd.DataFrame(parameter_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)
    validate_model_outputs(output, parameters, diagnostics)
    return output, parameters, diagnostics


def validate_model_outputs(
    data: pd.DataFrame,
    parameters: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> None:
    """Validate the final volatility panel and fitted-model summaries."""

    validate_returns(data)
    if len(parameters) != len(EXPECTED_MARKETS) or set(
        parameters["index_name"]
    ) != set(EXPECTED_MARKETS):
        raise ValueError("GARCH parameter table does not contain five markets.")
    if len(diagnostics) != len(EXPECTED_MARKETS) or set(
        diagnostics["index_name"]
    ) != set(EXPECTED_MARKETS):
        raise ValueError("GARCH diagnostics do not contain five markets.")
    numeric_parameters = parameters[
        ["mu", "omega", "alpha", "beta", "alpha_plus_beta", "nu"]
    ].to_numpy(dtype=float)
    if not np.isfinite(numeric_parameters).all():
        raise ValueError("GARCH parameter table contains a non-finite value.")
    if not np.allclose(
        parameters["alpha_plus_beta"],
        parameters["alpha"] + parameters["beta"],
        atol=RETURN_IDENTITY_TOLERANCE,
        rtol=0,
    ):
        raise ValueError("Saved GARCH persistence does not equal alpha plus beta.")
    if not diagnostics["converged"].all() or not diagnostics[
        "optimizer_success"
    ].all():
        raise ValueError("At least one saved GARCH diagnostic is not converged.")

    for index_name in EXPECTED_MARKETS:
        subset = data.loc[data["index_name"].eq(index_name)].reset_index(drop=True)
        missing_return = subset["log_return"].isna()
        missing_volatility = subset["garch_volatility"].isna()
        if not missing_volatility.equals(missing_return):
            raise ValueError(
                f"{index_name} volatility missingness does not match returns."
            )
        volatility = subset.loc[~missing_volatility, "garch_volatility"].to_numpy()
        if len(volatility) != EXPECTED_RETURN_COUNTS[index_name]:
            raise ValueError(f"{index_name} volatility length does not reconcile.")
        if not np.isfinite(volatility).all() or (volatility <= 0).any():
            raise ValueError(f"{index_name} volatility is not positive and finite.")
        nobs = int(
            parameters.loc[parameters["index_name"].eq(index_name), "nobs"].iloc[0]
        )
        if nobs != len(volatility):
            raise ValueError(f"{index_name} model nobs does not match volatility rows.")


def series_statistics(series: pd.Series) -> dict[str, float | int]:
    """Return stable descriptive statistics for one finite series."""

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


def extreme_date(data: pd.DataFrame, variable: str, kind: str) -> str:
    """Return the first ISO date at a variable's minimum or maximum."""

    valid = data.dropna(subset=[variable])
    position = valid[variable].idxmin() if kind == "minimum" else valid[variable].idxmax()
    return valid.loc[position, "date"].strftime("%Y-%m-%d")


def build_return_descriptives(data: pd.DataFrame) -> pd.DataFrame:
    """Describe decimal log returns and percentage-return estimation inputs."""

    rows: list[dict[str, Any]] = []
    definitions = [
        ("log_return", "decimal return"),
        ("return_pct", "percentage-return units"),
    ]
    for index_name in EXPECTED_MARKETS:
        subset = data.loc[data["index_name"].eq(index_name)]
        for variable, unit in definitions:
            rows.append(
                {
                    "index_name": index_name,
                    "variable": variable,
                    "unit": unit,
                    **series_statistics(subset[variable].dropna()),
                    "minimum_date": extreme_date(subset, variable, "minimum"),
                    "maximum_date": extreme_date(subset, variable, "maximum"),
                }
            )
    return pd.DataFrame(rows)


def build_volatility_descriptives(data: pd.DataFrame) -> pd.DataFrame:
    """Describe fitted conditional standard deviations by market."""

    rows: list[dict[str, Any]] = []
    for index_name in EXPECTED_MARKETS:
        subset = data.loc[data["index_name"].eq(index_name)]
        rows.append(
            {
                "index_name": index_name,
                "variable": "garch_volatility",
                "unit": "percentage-return standard-deviation units",
                **series_statistics(subset["garch_volatility"].dropna()),
                "minimum_date": extreme_date(
                    subset, "garch_volatility", "minimum"
                ),
                "maximum_date": extreme_date(
                    subset, "garch_volatility", "maximum"
                ),
            }
        )
    return pd.DataFrame(rows)


def configure_plots() -> None:
    """Apply the existing restrained thesis plotting style."""

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


def format_time_axis(ax: plt.Axes) -> None:
    """Use actual dates with readable annual and quarterly ticks."""

    ax.set_xlim(pd.Timestamp("2021-01-01"), pd.Timestamp("2023-12-31"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
    ax.tick_params(axis="x", which="minor", labelbottom=False)


def save_figure(fig: plt.Figure, path: Path) -> None:
    """Save one approximately 300-dpi PNG after layout is complete."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_market_series(
    data: pd.DataFrame,
    variable: str,
    title: str,
    y_label: str,
    path: Path,
    colors: dict[str, str],
) -> None:
    """Plot one unsmoothed market series in five vertically aligned panels."""

    fig, axes = plt.subplots(
        len(EXPECTED_MARKETS),
        1,
        figsize=(11, 11),
        sharex=True,
        sharey=True,
    )
    for ax, index_name in zip(axes, EXPECTED_MARKETS, strict=True):
        subset = data.loc[data["index_name"].eq(index_name)].dropna(
            subset=[variable]
        )
        ax.plot(
            subset["date"],
            subset[variable],
            color=colors[index_name],
            linewidth=0.75,
        )
        if variable == "log_return":
            ax.axhline(0, color="#4B5563", linewidth=0.7, linestyle="--")
        ax.set_title(index_name.replace("_", " "), loc="left", fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        format_time_axis(ax)
    axes[-1].set_xlabel("Trading date")
    fig.supylabel(y_label, x=0.015)
    fig.suptitle(title, fontsize=14)
    fig.text(
        0.5,
        0.012,
        "Observed trading dates only; no smoothing, interpolation, trimming, or normalization.",
        ha="center",
        fontsize=8.5,
        color="#374151",
    )
    fig.tight_layout(rect=[0.035, 0.03, 1, 0.97])
    save_figure(fig, path)


def build_figures(data: pd.DataFrame) -> None:
    """Build the two required five-market Phase 5 figures."""

    colors = {
        "EURO_STOXX_50": "#2563EB",
        "DAX": "#7C3AED",
        "CAC_40": "#0F766E",
        "FTSE_100": "#C2410C",
        "WIG20": "#B91C1C",
    }
    configure_plots()
    plot_market_series(
        data,
        "log_return",
        "Daily log returns across the five European equity indices",
        "Daily log return (decimal units)",
        MARKET_LOG_RETURNS_FIGURE,
        colors,
    )
    plot_market_series(
        data,
        "garch_volatility",
        "GARCH(1,1)-Student-t conditional volatility",
        "Conditional standard deviation (percentage-return units)",
        GARCH_CONDITIONAL_VOLATILITY_FIGURE,
        colors,
    )


def write_outputs(
    data: pd.DataFrame,
    return_descriptives: pd.DataFrame,
    parameters: pd.DataFrame,
    diagnostics: pd.DataFrame,
    volatility_descriptives: pd.DataFrame,
) -> None:
    """Write canonical data, tables, and diagnostics after all models validate."""

    outputs = {
        MARKET_RETURNS_GARCH_FILE: data,
        MARKET_RETURN_DESCRIPTIVES_TABLE: return_descriptives,
        GARCH_PARAMETERS_TABLE: parameters,
        GARCH_MODEL_DIAGNOSTICS_FILE: diagnostics,
        GARCH_VOLATILITY_DESCRIPTIVES_TABLE: volatility_descriptives,
    }
    for path, frame in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False, date_format="%Y-%m-%d", na_rep="")


def main() -> None:
    """Run Phase 5 only and leave all later empirical phases untouched."""

    validate_configuration()
    prices, input_sha256 = load_and_validate_prices()
    return_panel = calculate_returns(prices)
    output, parameters, diagnostics = estimate_garch_models(return_panel)
    return_descriptives = build_return_descriptives(output)
    volatility_descriptives = build_volatility_descriptives(output)
    write_outputs(
        output,
        return_descriptives,
        parameters,
        diagnostics,
        volatility_descriptives,
    )
    build_figures(output)

    if file_sha256(MARKET_PRICES_FILE) != input_sha256:
        raise RuntimeError("The immutable market-price input changed during Phase 5.")
    required_outputs = [
        MARKET_RETURNS_GARCH_FILE,
        MARKET_RETURN_DESCRIPTIVES_TABLE,
        GARCH_PARAMETERS_TABLE,
        GARCH_MODEL_DIAGNOSTICS_FILE,
        GARCH_VOLATILITY_DESCRIPTIVES_TABLE,
        MARKET_LOG_RETURNS_FIGURE,
        GARCH_CONDITIONAL_VOLATILITY_FIGURE,
    ]
    missing = [path for path in required_outputs if not path.exists()]
    empty = [path for path in required_outputs if path.exists() and path.stat().st_size == 0]
    if missing or empty:
        raise RuntimeError(f"Missing outputs: {missing}; empty outputs: {empty}.")

    print(f"Validated immutable market-price SHA-256: {input_sha256}")
    for index_name in EXPECTED_MARKETS:
        parameter = parameters.loc[parameters["index_name"].eq(index_name)].iloc[0]
        diagnostic = diagnostics.loc[
            diagnostics["index_name"].eq(index_name)
        ].iloc[0]
        print(
            f"{index_name}: prices={EXPECTED_PRICE_COUNTS[index_name]}, "
            f"returns={EXPECTED_RETURN_COUNTS[index_name]}, nobs={int(parameter['nobs'])}, "
            f"persistence={parameter['alpha_plus_beta']:.6f}, "
            f"nu={parameter['nu']:.6f}, converged={diagnostic['converged']}."
        )
    print(f"Saved canonical output to {MARKET_RETURNS_GARCH_FILE}")
    print(f"Canonical output SHA-256: {file_sha256(MARKET_RETURNS_GARCH_FILE)}")
    print(f"GARCH diagnostics SHA-256: {file_sha256(GARCH_MODEL_DIAGNOSTICS_FILE)}")
    print("Phase 5 market returns and GARCH volatility completed successfully.")


if __name__ == "__main__":
    main()
