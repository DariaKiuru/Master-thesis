"""Validate frozen Phase 4A scores and build the Phase 4B calendar series."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from config import (  # noqa: E402
    DAILY_REDDIT_FILE,
    DAILY_REDDIT_SUMMARY_FILE,
    END_DATE,
    FINBERT_REDDIT_FILE,
    START_DATE,
)


EXPECTED_INPUT_SHA256 = (
    "2E4A693558197B8007F81C5D348362140524E3C31A8D31293F86D691DDB9C7FF"
)
EXPECTED_POSTS = 1_503
EXPECTED_LABEL_COUNTS = {"positive": 106, "neutral": 953, "negative": 444}
EXPECTED_ANNUAL_POST_COUNTS = {2021: 83, 2022: 1_197, 2023: 223}
EXPECTED_CALENDAR_DAYS = 1_095
PROBABILITY_TOLERANCE = 1e-6
SCORE_TOLERANCE = 1e-12
PROBABILITY_COLUMNS = [
    "post_positive_probability",
    "post_neutral_probability",
    "post_negative_probability",
]
REQUIRED_COLUMNS = {
    "id",
    "date_utc",
    "subreddit",
    *PROBABILITY_COLUMNS,
    "sentiment_score",
    "sentiment_label",
}
SENTIMENT_LABELS = ["positive", "neutral", "negative"]
COUNT_COLUMNS = [
    "post_count",
    "positive_post_count",
    "neutral_post_count",
    "negative_post_count",
]
MEAN_PROBABILITY_COLUMNS = [
    "mean_positive_probability",
    "mean_neutral_probability",
    "mean_negative_probability",
]
DAILY_COLUMNS = [
    "date",
    "sentiment",
    "post_count",
    "positive_post_count",
    "neutral_post_count",
    "negative_post_count",
    *MEAN_PROBABILITY_COLUMNS,
]


def calculate_sha256(path: Path) -> str:
    """Return an uppercase SHA-256 digest without loading the file at once."""

    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def load_post_scores() -> tuple[pd.DataFrame, str]:
    """Load the genuine final Phase 4A output, never a diagnostic sample."""

    if not FINBERT_REDDIT_FILE.exists():
        raise FileNotFoundError(
            "Final Phase 4A output does not exist yet; daily aggregation was not run: "
            f"{FINBERT_REDDIT_FILE}"
        )
    input_sha256 = calculate_sha256(FINBERT_REDDIT_FILE)
    if input_sha256 != EXPECTED_INPUT_SHA256:
        raise ValueError(
            "Canonical Phase 4A SHA-256 changed; Phase 4B aggregation stopped. "
            f"Expected {EXPECTED_INPUT_SHA256}, found {input_sha256}."
        )

    data = pd.read_csv(
        FINBERT_REDDIT_FILE,
        dtype={"id": "string", "subreddit": "string"},
    )
    if missing := REQUIRED_COLUMNS - set(data.columns):
        raise ValueError(f"Final Phase 4A output lacks columns: {sorted(missing)}")
    return data, input_sha256


def validate_post_scores(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    """Validate post membership, missingness, probabilities, scores, and labels."""

    if len(data) != EXPECTED_POSTS:
        raise ValueError(f"Expected 1,503 Phase 4A posts, found {len(data):,}.")
    if data["id"].isna().any() or data["id"].astype(str).str.strip().eq("").any():
        raise ValueError("A Phase 4A post has a missing or blank Reddit ID.")
    duplicate_ids = int(data["id"].duplicated().sum())
    unique_ids = int(data["id"].nunique())
    if duplicate_ids or unique_ids != EXPECTED_POSTS:
        raise ValueError("Phase 4A post IDs must be globally unique.")
    for column in REQUIRED_COLUMNS:
        if data[column].isna().any():
            raise ValueError(f"Required Phase 4A column {column} contains missing values.")
        if pd.api.types.is_object_dtype(data[column]) or pd.api.types.is_string_dtype(
            data[column]
        ):
            if data[column].astype(str).str.strip().eq("").any():
                raise ValueError(f"Required Phase 4A column {column} contains blanks.")

    validated = data.copy()
    dates = pd.to_datetime(validated["date_utc"], errors="coerce")
    if dates.isna().any() or not dates.between(
        START_DATE, END_DATE, inclusive="both"
    ).all():
        raise ValueError("Phase 4A output contains an invalid or out-of-period date.")
    validated["date"] = dates.dt.normalize()

    for column in [*PROBABILITY_COLUMNS, "sentiment_score"]:
        validated[column] = pd.to_numeric(validated[column], errors="coerce")
    numeric_values = validated[[*PROBABILITY_COLUMNS, "sentiment_score"]].to_numpy(
        dtype=float
    )
    if not np.isfinite(numeric_values).all():
        raise ValueError("Phase 4A probabilities or scores contain non-finite values.")
    probabilities = validated[PROBABILITY_COLUMNS].to_numpy(dtype=float)
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Phase 4A post probabilities fall outside [0, 1].")
    probability_sum_error = np.abs(probabilities.sum(axis=1) - 1.0)
    max_probability_sum_error = float(probability_sum_error.max())
    if max_probability_sum_error > PROBABILITY_TOLERANCE:
        raise ValueError("Phase 4A post probabilities do not sum to one.")

    scores = validated["sentiment_score"].to_numpy(dtype=float)
    if ((scores < -1) | (scores > 1)).any():
        raise ValueError("Phase 4A sentiment scores fall outside [-1, 1].")
    expected_scores = probabilities[:, 0] - probabilities[:, 2]
    score_identity_error = np.abs(scores - expected_scores)
    max_score_identity_error = float(score_identity_error.max())
    if max_score_identity_error > SCORE_TOLERANCE:
        raise ValueError("sentiment_score is not positive probability minus negative.")

    labels = validated["sentiment_label"].astype(str)
    if set(labels.unique()) - set(SENTIMENT_LABELS):
        raise ValueError("Phase 4A output contains an invalid sentiment label.")
    expected_labels = np.asarray(SENTIMENT_LABELS)[probabilities.argmax(axis=1)]
    if not np.array_equal(labels.to_numpy(), expected_labels):
        raise ValueError("Phase 4A sentiment labels do not equal probability argmax.")

    label_counts = labels.value_counts().reindex(SENTIMENT_LABELS, fill_value=0)
    if label_counts.to_dict() != EXPECTED_LABEL_COUNTS:
        raise ValueError(
            "Phase 4A label totals do not reconcile to the frozen expectations: "
            f"{label_counts.to_dict()}."
        )
    annual_counts = (
        validated.groupby(validated["date"].dt.year).size().astype(int).to_dict()
    )
    if annual_counts != EXPECTED_ANNUAL_POST_COUNTS:
        raise ValueError(
            "Phase 4A annual post totals do not reconcile to the frozen "
            f"expectations: {annual_counts}."
        )

    diagnostics = {
        "rows": len(validated),
        "unique_ids": unique_ids,
        "duplicate_ids": duplicate_ids,
        "max_probability_sum_error": max_probability_sum_error,
        "max_score_identity_error": max_score_identity_error,
        "minimum_sentiment_score": float(scores.min()),
        "maximum_sentiment_score": float(scores.max()),
        "label_counts": label_counts.to_dict(),
        "annual_counts": annual_counts,
    }
    return validated, diagnostics


def build_daily_series(posts: pd.DataFrame) -> pd.DataFrame:
    """Calculate the complete calendar-day sentiment and attention series."""

    observed = (
        posts.groupby("date", sort=True)
        .agg(
            sentiment=("sentiment_score", "mean"),
            post_count=("id", "size"),
            positive_post_count=(
                "sentiment_label",
                lambda labels: labels.eq("positive").sum(),
            ),
            neutral_post_count=(
                "sentiment_label",
                lambda labels: labels.eq("neutral").sum(),
            ),
            negative_post_count=(
                "sentiment_label",
                lambda labels: labels.eq("negative").sum(),
            ),
            mean_positive_probability=("post_positive_probability", "mean"),
            mean_neutral_probability=("post_neutral_probability", "mean"),
            mean_negative_probability=("post_negative_probability", "mean"),
        )
        .reset_index()
    )
    calendar = pd.DataFrame(
        {"date": pd.date_range(START_DATE, END_DATE, freq="D")}
    )
    daily = calendar.merge(observed, on="date", how="left", validate="one_to_one")
    daily[COUNT_COLUMNS] = daily[COUNT_COLUMNS].fillna(0).astype("int64")
    daily = daily.loc[:, DAILY_COLUMNS]
    return daily


def validate_daily_series(daily: pd.DataFrame) -> dict[str, object]:
    """Confirm full calendar coverage and preserve missing no-discussion sentiment."""

    if daily.empty or list(daily.columns) != DAILY_COLUMNS:
        raise ValueError("The daily Reddit series is empty or has an invalid schema.")
    if daily["date"].duplicated().any() or not daily["date"].is_monotonic_increasing:
        raise ValueError("Daily Reddit dates must be unique and sorted.")
    expected_calendar = pd.date_range(START_DATE, END_DATE, freq="D")
    if len(daily) != EXPECTED_CALENDAR_DAYS or len(daily) != len(expected_calendar):
        raise ValueError("Daily Reddit output does not contain 1,095 calendar days.")
    if not np.array_equal(
        daily["date"].to_numpy(), expected_calendar.to_numpy()
    ):
        raise ValueError("Daily Reddit output does not span the complete sample calendar.")
    attention = daily["post_count"].to_numpy(dtype=float)
    if (
        not np.isfinite(attention).all()
        or (attention < 0).any()
        or not np.equal(attention, np.floor(attention)).all()
    ):
        raise ValueError("Daily attention must contain non-negative integer post counts.")
    if int(attention.sum()) != EXPECTED_POSTS:
        raise ValueError("Daily attention does not reconcile to post-level observations.")
    discussion_days = daily["post_count"].gt(0)
    sentiment = pd.to_numeric(daily["sentiment"], errors="coerce")
    if sentiment.loc[discussion_days].isna().any():
        raise ValueError("A discussion day is missing its mean sentiment.")
    observed_sentiment = sentiment.loc[discussion_days].to_numpy(dtype=float)
    if not np.isfinite(observed_sentiment).all() or (
        (observed_sentiment < -1) | (observed_sentiment > 1)
    ).any():
        raise ValueError("Observed daily sentiment is non-finite or outside [-1, 1].")
    if sentiment.loc[~discussion_days].notna().any():
        raise ValueError("A no-discussion day was assigned a sentiment value.")

    if not daily.loc[~discussion_days, COUNT_COLUMNS[1:]].eq(0).all().all():
        raise ValueError("A no-discussion day has a non-zero class count.")
    if daily.loc[~discussion_days, MEAN_PROBABILITY_COLUMNS].notna().any().any():
        raise ValueError("A no-discussion day has a mean probability value.")

    class_count_sum = daily[COUNT_COLUMNS[1:]].sum(axis=1)
    if not class_count_sum.eq(daily["post_count"]).all():
        raise ValueError("Daily class counts do not sum to daily post_count.")
    class_totals = {
        label: int(daily[f"{label}_post_count"].sum())
        for label in SENTIMENT_LABELS
    }
    if class_totals != EXPECTED_LABEL_COUNTS:
        raise ValueError(
            f"Daily class totals do not match Phase 4A: {class_totals}."
        )

    populated_probabilities = daily.loc[
        discussion_days, MEAN_PROBABILITY_COLUMNS
    ].to_numpy(dtype=float)
    if not np.isfinite(populated_probabilities).all():
        raise ValueError("A populated day has a missing or non-finite mean probability.")
    if (
        (populated_probabilities < 0) | (populated_probabilities > 1)
    ).any():
        raise ValueError("A populated day has a mean probability outside [0, 1].")
    daily_probability_sum_error = np.abs(populated_probabilities.sum(axis=1) - 1)
    max_daily_probability_sum_error = float(daily_probability_sum_error.max())
    if max_daily_probability_sum_error > PROBABILITY_TOLERANCE:
        raise ValueError("Daily mean probabilities do not sum to one.")
    daily_identity_error = np.abs(
        observed_sentiment
        - (populated_probabilities[:, 0] - populated_probabilities[:, 2])
    )
    max_daily_identity_error = float(daily_identity_error.max())
    if max_daily_identity_error > SCORE_TOLERANCE:
        raise ValueError(
            "Daily sentiment is not mean positive minus mean negative probability."
        )

    annual_counts = (
        daily.groupby(daily["date"].dt.year)["post_count"].sum().astype(int).to_dict()
    )
    if annual_counts != EXPECTED_ANNUAL_POST_COUNTS:
        raise ValueError(
            f"Daily annual post totals do not reconcile to Phase 4A: {annual_counts}."
        )

    return {
        "calendar_days": len(daily),
        "duplicate_dates": int(daily["date"].duplicated().sum()),
        "dates_monotonic": bool(daily["date"].is_monotonic_increasing),
        "discussion_days": int(discussion_days.sum()),
        "zero_post_days": int((~discussion_days).sum()),
        "total_post_count": int(daily["post_count"].sum()),
        "class_totals": class_totals,
        "annual_counts": annual_counts,
        "minimum_daily_sentiment": float(observed_sentiment.min()),
        "maximum_daily_sentiment": float(observed_sentiment.max()),
        "max_daily_probability_sum_error": max_daily_probability_sum_error,
        "max_daily_identity_error": max_daily_identity_error,
        "invalid_daily_sentiment_observations": 0,
    }


def describe_series(series: pd.Series) -> dict[str, float | int]:
    """Return the requested descriptive statistics with stable metric names."""

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


def matching_dates(daily: pd.DataFrame, column: str, value: float) -> str:
    """Return semicolon-delimited ISO dates matching a diagnostic extreme."""

    return ";".join(
        daily.loc[daily[column].eq(value), "date"].dt.strftime("%Y-%m-%d")
    )


def build_summary(
    daily: pd.DataFrame,
    input_sha256: str,
    post_diagnostics: dict[str, object],
    daily_diagnostics: dict[str, object],
) -> pd.DataFrame:
    """Create one tidy reproducibility summary for Phase 4B."""

    records: list[dict[str, object]] = []

    def add(section: str, metric: str, value: object, period: str = "all") -> None:
        records.append(
            {"section": section, "period": period, "metric": metric, "value": value}
        )

    add(
        "input_validation",
        "input_path",
        FINBERT_REDDIT_FILE.relative_to(REPOSITORY_ROOT).as_posix(),
    )
    add("input_validation", "sha256", input_sha256)
    add("input_validation", "sha256_matches_frozen_phase_4a", True)
    add("input_validation", "required_fields_present", True)
    add("input_validation", "rows", post_diagnostics["rows"])
    add("input_validation", "unique_ids", post_diagnostics["unique_ids"])
    add("input_validation", "duplicate_ids", post_diagnostics["duplicate_ids"])
    add("input_validation", "probability_tolerance", PROBABILITY_TOLERANCE)
    add("input_validation", "score_identity_tolerance", SCORE_TOLERANCE)
    add(
        "input_validation",
        "maximum_probability_sum_error",
        post_diagnostics["max_probability_sum_error"],
    )
    add(
        "input_validation",
        "maximum_score_identity_error",
        post_diagnostics["max_score_identity_error"],
    )
    for label in SENTIMENT_LABELS:
        add(
            "input_validation",
            f"{label}_label_count",
            post_diagnostics["label_counts"][label],
        )

    add("calendar_reconciliation", "calendar_days", len(daily))
    add("calendar_reconciliation", "start_date", daily["date"].min().date())
    add("calendar_reconciliation", "end_date", daily["date"].max().date())
    add(
        "calendar_reconciliation",
        "duplicate_dates",
        daily_diagnostics["duplicate_dates"],
    )
    add(
        "calendar_reconciliation",
        "dates_monotonic_increasing",
        daily_diagnostics["dates_monotonic"],
    )
    add(
        "calendar_reconciliation",
        "days_with_sentiment",
        daily_diagnostics["discussion_days"],
    )
    add(
        "calendar_reconciliation",
        "zero_post_days",
        daily_diagnostics["zero_post_days"],
    )
    add(
        "post_reconciliation",
        "sum_post_count",
        daily_diagnostics["total_post_count"],
    )
    for label in SENTIMENT_LABELS:
        add(
            "class_count_reconciliation",
            f"sum_{label}_post_count",
            daily_diagnostics["class_totals"][label],
        )
    add(
        "probability_identities",
        "maximum_mean_probability_sum_error",
        daily_diagnostics["max_daily_probability_sum_error"],
    )
    add(
        "probability_identities",
        "maximum_sentiment_identity_error",
        daily_diagnostics["max_daily_identity_error"],
    )
    add(
        "probability_identities",
        "invalid_daily_sentiment_observations",
        daily_diagnostics["invalid_daily_sentiment_observations"],
    )

    sentiment_description = describe_series(daily["sentiment"].dropna())
    post_count_description = describe_series(daily["post_count"])
    for metric, value in sentiment_description.items():
        add("sentiment_descriptives", metric, value)
    for metric, value in post_count_description.items():
        add("post_count_descriptives", metric, value)

    annual = daily.groupby(daily["date"].dt.year).agg(
        sentiment_observation_days=("sentiment", "count"),
        post_count=("post_count", "sum"),
        mean_sentiment=("sentiment", "mean"),
    )
    for year, row in annual.iterrows():
        add(
            "annual_summary",
            "sentiment_observation_days",
            int(row["sentiment_observation_days"]),
            str(year),
        )
        add("annual_summary", "post_count", int(row["post_count"]), str(year))
        add(
            "annual_summary",
            "mean_sentiment",
            float(row["mean_sentiment"]),
            str(year),
        )

    maximum_activity = int(daily["post_count"].max())
    minimum_sentiment = float(daily["sentiment"].min())
    maximum_sentiment = float(daily["sentiment"].max())
    add("extremes", "maximum_daily_post_count", maximum_activity)
    add(
        "extremes",
        "maximum_daily_post_count_dates",
        matching_dates(daily, "post_count", maximum_activity),
    )
    add("extremes", "minimum_observed_daily_sentiment", minimum_sentiment)
    add(
        "extremes",
        "minimum_observed_daily_sentiment_dates",
        matching_dates(daily, "sentiment", minimum_sentiment),
    )
    add("extremes", "maximum_observed_daily_sentiment", maximum_sentiment)
    add(
        "extremes",
        "maximum_observed_daily_sentiment_dates",
        matching_dates(daily, "sentiment", maximum_sentiment),
    )
    return pd.DataFrame(records, columns=["section", "period", "metric", "value"])


def write_outputs(daily: pd.DataFrame, summary: pd.DataFrame) -> None:
    """Write only calendar-day Reddit outputs after every validation passes."""

    DAILY_REDDIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    DAILY_REDDIT_SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(
        DAILY_REDDIT_FILE,
        index=False,
        date_format="%Y-%m-%d",
        na_rep="",
    )
    summary.to_csv(DAILY_REDDIT_SUMMARY_FILE, index=False)


def main() -> None:
    """Build calendar-day sentiment and attention from genuine Phase 4A data."""

    raw_posts, input_sha256 = load_post_scores()
    posts, post_diagnostics = validate_post_scores(raw_posts)
    daily = build_daily_series(posts)
    daily_diagnostics = validate_daily_series(daily)
    summary = build_summary(
        daily,
        input_sha256,
        post_diagnostics,
        daily_diagnostics,
    )
    write_outputs(daily, summary)
    print(f"Validated {len(posts):,} post-level FinBERT observations.")
    print(
        f"Built {len(daily):,} complete calendar-day rows: "
        f"{daily_diagnostics['discussion_days']:,} with sentiment and "
        f"{daily_diagnostics['zero_post_days']:,} with zero posts."
    )
    print(
        "Reconciled labels: "
        + ", ".join(
            f"{label}={daily_diagnostics['class_totals'][label]:,}"
            for label in SENTIMENT_LABELS
        )
        + "."
    )
    print(f"Saved daily Reddit series to {DAILY_REDDIT_FILE}")
    print(f"Saved daily aggregation summary to {DAILY_REDDIT_SUMMARY_FILE}")


if __name__ == "__main__":
    main()
