"""Phase 3B final: build the cleaned, FinBERT-ready Reddit sample.

Why this phase exists
    The broad candidate corpus must be restricted to English-language,
    Ukraine-war-related financial discussion before measuring sentiment.

Main inputs
    The 3,033-post Phase 3A corpus, frozen final relevance vocabularies, and the
    150-post manual language-validation sample.

Main outputs
    ``data/processed/reddit_posts_cleaned.csv`` containing the final 1,503
    qualifying posts, plus cleaning and language diagnostics.

Methodological rules and boundaries
    Only r/investing, r/stocks, and r/StockMarket are eligible. Available title
    and body text are combined; unavailable bodies use the title alone. The
    final relevance logic requires crisis-specific financial discussion, and
    confidently non-English posts are excluded. Each retained post remains one
    observation and later receives equal empirical weight. This script does
    not run or fine-tune FinBERT and does not aggregate posts by day.
"""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from langdetect import DetectorFactory, detect_langs
from langdetect.lang_detect_exception import LangDetectException


# Allow ``python src/06_clean_reddit.py`` from the repository root.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from config import (  # noqa: E402
    CLEANED_REDDIT_FILE,
    END_DATE,
    RAW_REDDIT_FILE,
    REDDIT_CLEANING_SUMMARY_FILE,
    REDDIT_DIRECT_UKRAINE_CONTEXT_TERMS,
    REDDIT_EXTRACTION_KEYWORDS,
    REDDIT_FINAL_CONFLICT_CONTEXT_TERMS,
    REDDIT_FINAL_CRISIS_FINANCIAL_CONSEQUENCE_PHRASES,
    REDDIT_LANGUAGE_CONFIDENT_ENGLISH_PROBABILITY,
    REDDIT_LANGUAGE_CONFIDENT_NON_ENGLISH_MIN_WORDS,
    REDDIT_LANGUAGE_CONFIDENT_NON_ENGLISH_PROBABILITY,
    REDDIT_LANGUAGE_DETECTOR_SEED,
    REDDIT_LANGUAGE_MIN_ALPHA_CHARACTERS,
    REDDIT_LANGUAGE_MIN_ALPHA_WORDS,
    REDDIT_LANGUAGE_VALIDATION_FILE,
    REDDIT_MAX_EXPECTED_FINAL_RELEVANCE_INCREASE,
    REDDIT_REFINED_CONFLICT_CONTEXT_TERMS,
    REDDIT_REFINED_CRISIS_FINANCIAL_CONSEQUENCE_PHRASES,
    REDDIT_REFINED_FINANCIAL_CONTEXT_TERMS,
    REDDIT_RELEVANCE_REVIEW_SAMPLE_FILE,
    REDDIT_RUSSIA_CONTEXT_TERMS,
    START_DATE,
    SUBREDDITS,
)


EXPECTED_RAW_POSTS = 3_033
EXPECTED_PHASE3B3_RETAINED = 1_491
EXPECTED_REVIEW_SAMPLE_POSTS = 150
UNAVAILABLE_BODY_VALUES = {"", "[removed]", "[deleted]"}

CLEANED_COLUMNS = [
    "id",
    "created_utc",
    "created_datetime_utc",
    "date_utc",
    "subreddit",
    "title",
    "selftext",
    "full_text",
    "finbert_text",
    "score",
    "num_comments",
    "url",
    "extraction_matched_keywords_original",
    "matched_keywords_recomputed",
    "body_status",
    "title_only",
    "has_direct_ukraine_context",
    "has_russia_context",
    "has_conflict_context",
    "has_financial_context",
    "has_crisis_financial_consequence",
    "geopolitical_crisis_financial",
    "crisis_specific_financial_consequence",
    "relevant_candidate",
    "relevance_path",
    "language_status",
    "detected_language",
    "language_probability",
]

LANGUAGE_VALIDATION_COLUMNS = [
    "id",
    "date_utc",
    "subreddit",
    "title",
    "selftext",
    "review_group",
    "full_text",
    "body_status",
    "title_only",
    "alphabetic_word_count",
    "alphabetic_character_count",
    "language_status",
    "detected_language",
    "language_probability",
]


# ---------------------------------------------------------------------------
# Reproduce the frozen relevance rule without altering source text
# ---------------------------------------------------------------------------

def file_sha256(path: Path) -> str:
    """Calculate a file checksum for immutability and reproducibility checks."""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bounded_pattern(term: str) -> str:
    """Escape a term while allowing flexible whitespace within phrases."""

    return re.escape(term).replace(r"\ ", r"\s+")


def compile_term_pattern(terms: list[str]) -> re.Pattern[str]:
    """Compile case-insensitive terms with non-word boundaries."""

    if not terms:
        raise ValueError("A relevance vocabulary cannot be empty.")
    alternatives = "|".join(
        bounded_pattern(term) for term in sorted(terms, key=len, reverse=True)
    )
    return re.compile(rf"(?<!\w)(?:{alternatives})(?!\w)", re.IGNORECASE)


EXTRACTION_PATTERNS = {
    keyword: compile_term_pattern([keyword])
    for keyword in REDDIT_EXTRACTION_KEYWORDS
}
DIRECT_UKRAINE_PATTERN = compile_term_pattern(
    REDDIT_DIRECT_UKRAINE_CONTEXT_TERMS
)
RUSSIA_CONTEXT_PATTERN = compile_term_pattern(REDDIT_RUSSIA_CONTEXT_TERMS)
PHASE3B3_CONFLICT_PATTERN = compile_term_pattern(
    REDDIT_REFINED_CONFLICT_CONTEXT_TERMS
)
FINAL_CONFLICT_PATTERN = compile_term_pattern(
    REDDIT_FINAL_CONFLICT_CONTEXT_TERMS
)
FINANCIAL_PATTERN = compile_term_pattern(
    REDDIT_REFINED_FINANCIAL_CONTEXT_TERMS
)
PHASE3B3_CONSEQUENCE_PATTERN = compile_term_pattern(
    REDDIT_REFINED_CRISIS_FINANCIAL_CONSEQUENCE_PHRASES
)
FINAL_CONSEQUENCE_PATTERN = compile_term_pattern(
    REDDIT_FINAL_CRISIS_FINANCIAL_CONSEQUENCE_PHRASES
)
URL_PATTERN = re.compile(
    r"(?i)(?<!\S)(?:https?://|www\.)\S+",
)
ALPHABETIC_WORD_PATTERN = re.compile(r"[^\W\d_]+", re.UNICODE)


def validate_configuration() -> None:
    """Ensure the frozen rule differs only by the approved narrow additions."""

    if SUBREDDITS != ["investing", "stocks", "StockMarket"]:
        raise ValueError("The configured subreddit universe is incorrect.")
    if START_DATE != "2021-01-01" or END_DATE != "2023-12-31":
        raise ValueError("The configured sample period is incorrect.")
    if "crises" not in REDDIT_FINAL_CONFLICT_CONTEXT_TERMS:
        raise ValueError("The final conflict vocabulary must include crises.")
    if set(REDDIT_REFINED_CONFLICT_CONTEXT_TERMS) - set(
        REDDIT_FINAL_CONFLICT_CONTEXT_TERMS
    ):
        raise ValueError("The final rule dropped a Phase 3B.3 conflict term.")
    if set(REDDIT_REFINED_CRISIS_FINANCIAL_CONSEQUENCE_PHRASES) - set(
        REDDIT_FINAL_CRISIS_FINANCIAL_CONSEQUENCE_PHRASES
    ):
        raise ValueError("The final rule dropped a Phase 3B.3 consequence.")

    approved_examples = [
        "cut off from SWIFT",
        "cut Russia off from SWIFT",
        "kicked off SWIFT",
        "taken off SWIFT",
        "trading halted",
        "ADR conversion",
        "Russian securities suspended",
        "freeze funds exposed to Russian assets",
    ]
    if any(
        not FINAL_CONSEQUENCE_PATTERN.search(text)
        for text in approved_examples
    ):
        raise ValueError("An approved final consequence variant is missing.")


def load_raw_corpus() -> pd.DataFrame:
    """Load and validate the immutable Phase 3A candidate corpus."""

    raw = pd.read_csv(
        RAW_REDDIT_FILE,
        dtype={"id": "string", "subreddit": "string"},
        keep_default_na=False,
    )
    required = {
        "id",
        "created_utc",
        "created_datetime_utc",
        "date_utc",
        "subreddit",
        "title",
        "selftext",
        "score",
        "num_comments",
        "url",
        "matched_search_keywords",
    }
    if missing := required - set(raw.columns):
        raise ValueError(f"Raw Reddit corpus lacks columns: {sorted(missing)}")
    if len(raw) != EXPECTED_RAW_POSTS or raw["id"].duplicated().any():
        raise ValueError("Expected exactly 3,033 unique Phase 3A posts.")
    if set(raw["subreddit"].unique()) != set(SUBREDDITS):
        raise ValueError("Raw corpus contains an unexpected subreddit.")

    raw["created_datetime_utc"] = pd.to_datetime(
        raw["created_datetime_utc"], errors="coerce", utc=True
    )
    if raw["created_datetime_utc"].isna().any():
        raise ValueError("A raw UTC timestamp could not be parsed.")
    expected_dates = raw["created_datetime_utc"].dt.strftime("%Y-%m-%d")
    if not expected_dates.equals(raw["date_utc"].astype(str)):
        raise ValueError("date_utc is inconsistent with created_datetime_utc.")
    if not raw["date_utc"].between(START_DATE, END_DATE).all():
        raise ValueError("Raw corpus contains an out-of-period date.")
    return raw


def body_status(selftext: object) -> str:
    """Classify body availability without changing the source selftext."""

    stripped = str(selftext).strip()
    lowered = stripped.casefold()
    if stripped == "":
        return "blank"
    if lowered == "[removed]":
        return "removed"
    if lowered == "[deleted]":
        return "deleted"
    return "available"


def add_usable_text(data: pd.DataFrame) -> pd.DataFrame:
    """Use title plus available body, or title alone for unavailable bodies."""

    result = data.copy()
    result["body_status"] = result["selftext"].map(body_status)
    result["title_only"] = result["body_status"].ne("available")
    result["full_text"] = np.where(
        result["title_only"],
        result["title"].astype(str),
        result["title"].astype(str) + " " + result["selftext"].astype(str),
    )
    return result


def recompute_extraction_keywords(text: str) -> str:
    """Recompute extraction provenance with whole-word matching."""

    return "|".join(
        keyword
        for keyword in REDDIT_EXTRACTION_KEYWORDS
        if EXTRACTION_PATTERNS[keyword].search(text)
    )


def relevance_path(
    geopolitical: pd.Series,
    consequence: pd.Series,
) -> pd.Series:
    """Label the final transparent inclusion path."""

    path_count = geopolitical.astype(int) + consequence.astype(int)
    return pd.Series(
        np.select(
            [path_count.eq(2), geopolitical, consequence],
            [
                "multiple_paths",
                "geopolitical_crisis_financial",
                "crisis_specific_financial_consequence",
            ],
            default="excluded",
        ),
        index=geopolitical.index,
        dtype="string",
    )


def apply_relevance_rules(data: pd.DataFrame) -> pd.DataFrame:
    """Reproduce Phase 3B.3 and apply only the approved final corrections."""

    result = data.copy()
    text = result["full_text"]
    result["extraction_matched_keywords_original"] = result[
        "matched_search_keywords"
    ]
    result["matched_keywords_recomputed"] = text.map(
        recompute_extraction_keywords
    )
    result["has_direct_ukraine_context"] = text.str.contains(
        DIRECT_UKRAINE_PATTERN, na=False
    )
    result["has_russia_context"] = text.str.contains(
        RUSSIA_CONTEXT_PATTERN, na=False
    )
    result["has_financial_context"] = text.str.contains(
        FINANCIAL_PATTERN, na=False
    )

    shared_context = (
        result["has_direct_ukraine_context"]
        | result["has_russia_context"]
    )
    phase3b3_conflict = text.str.contains(
        PHASE3B3_CONFLICT_PATTERN, na=False
    )
    phase3b3_consequence = text.str.contains(
        PHASE3B3_CONSEQUENCE_PATTERN, na=False
    )
    phase3b3_geopolitical = (
        shared_context
        & phase3b3_conflict
        & result["has_financial_context"]
    )
    phase3b3_consequence_path = shared_context & phase3b3_consequence
    result["phase3b3_relevant_candidate"] = (
        phase3b3_geopolitical | phase3b3_consequence_path
    )

    result["has_conflict_context"] = text.str.contains(
        FINAL_CONFLICT_PATTERN, na=False
    )
    result["has_crisis_financial_consequence"] = text.str.contains(
        FINAL_CONSEQUENCE_PATTERN, na=False
    )
    result["geopolitical_crisis_financial"] = (
        shared_context
        & result["has_conflict_context"]
        & result["has_financial_context"]
    )
    result["crisis_specific_financial_consequence"] = (
        shared_context & result["has_crisis_financial_consequence"]
    )
    result["relevant_candidate"] = (
        result["geopolitical_crisis_financial"]
        | result["crisis_specific_financial_consequence"]
    )
    result["relevance_path"] = relevance_path(
        result["geopolitical_crisis_financial"],
        result["crisis_specific_financial_consequence"],
    )
    return result


def validate_relevance(data: pd.DataFrame) -> int:
    """Confirm the frozen formula and guard against an unexpectedly large change."""

    phase3b3_count = int(data["phase3b3_relevant_candidate"].sum())
    final_count = int(data["relevant_candidate"].sum())
    if phase3b3_count != EXPECTED_PHASE3B3_RETAINED:
        raise ValueError(
            f"Phase 3B.3 did not reproduce 1,491 posts: {phase3b3_count}."
        )
    if (
        data["phase3b3_relevant_candidate"]
        & ~data["relevant_candidate"]
    ).any():
        raise ValueError("A final vocabulary addition removed a Phase 3B.3 post.")
    increase = final_count - phase3b3_count
    if increase < 0 or increase > REDDIT_MAX_EXPECTED_FINAL_RELEVANCE_INCREASE:
        raise RuntimeError(
            "Unexpected final relevance change: "
            f"{increase:+,} posts relative to Phase 3B.3."
        )

    expected = (
        (
            data["has_direct_ukraine_context"]
            | data["has_russia_context"]
        )
        & data["has_conflict_context"]
        & data["has_financial_context"]
    ) | (
        (
            data["has_direct_ukraine_context"]
            | data["has_russia_context"]
        )
        & data["has_crisis_financial_consequence"]
    )
    if not np.array_equal(expected, data["relevant_candidate"]):
        raise ValueError("The final relevance formula is incorrect.")
    if recompute_extraction_keywords("Russian stocks") != "Russian":
        raise ValueError("Russian is still being confused with Russia.")
    if recompute_extraction_keywords("Russia stocks") != "Russia":
        raise ValueError("Russia is not matched independently.")
    return increase


def language_result(text: str) -> tuple[str, str, float, int, int]:
    """Return a conservative deterministic language status and diagnostics."""

    words = ALPHABETIC_WORD_PATTERN.findall(text)
    alpha_word_count = len(words)
    alpha_character_count = sum(character.isalpha() for character in text)
    if (
        alpha_word_count < REDDIT_LANGUAGE_MIN_ALPHA_WORDS
        or alpha_character_count < REDDIT_LANGUAGE_MIN_ALPHA_CHARACTERS
    ):
        return (
            "uncertain_short",
            "",
            np.nan,
            alpha_word_count,
            alpha_character_count,
        )
    try:
        best = detect_langs(text)[0]
    except LangDetectException:
        return (
            "uncertain",
            "",
            np.nan,
            alpha_word_count,
            alpha_character_count,
        )
    language = str(best.lang)
    probability = float(best.prob)
    if (
        language == "en"
        and probability >= REDDIT_LANGUAGE_CONFIDENT_ENGLISH_PROBABILITY
    ):
        status = "detected_english"
    elif (
        language != "en"
        and probability
        >= REDDIT_LANGUAGE_CONFIDENT_NON_ENGLISH_PROBABILITY
        and alpha_word_count
        >= REDDIT_LANGUAGE_CONFIDENT_NON_ENGLISH_MIN_WORDS
    ):
        status = "detected_non_english"
    else:
        status = "uncertain"
    return (
        status,
        language,
        probability,
        alpha_word_count,
        alpha_character_count,
    )


def add_language_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Apply deterministic conservative language classification."""

    result = data.copy()
    language_rows = [language_result(text) for text in result["full_text"]]
    language = pd.DataFrame(
        language_rows,
        columns=[
            "language_status",
            "detected_language",
            "language_probability",
            "alphabetic_word_count",
            "alphabetic_character_count",
        ],
        index=result.index,
    )
    return pd.concat([result, language], axis="columns")


def validate_manual_language_sample() -> pd.DataFrame:
    """Validate the detector against the 150-post manually reviewed sample."""

    sample = pd.read_csv(
        REDDIT_RELEVANCE_REVIEW_SAMPLE_FILE,
        dtype={"id": "string", "subreddit": "string"},
        keep_default_na=False,
    )
    required = {
        "id",
        "date_utc",
        "subreddit",
        "title",
        "selftext",
        "review_group",
    }
    if missing := required - set(sample.columns):
        raise ValueError(f"Manual review sample lacks columns: {sorted(missing)}")
    if (
        len(sample) != EXPECTED_REVIEW_SAMPLE_POSTS
        or sample["id"].duplicated().any()
    ):
        raise ValueError("Expected the 150 unique manually reviewed posts.")
    sample = add_usable_text(sample)
    sample = add_language_columns(sample)
    flagged = sample["language_status"].eq("detected_non_english")
    if flagged.any():
        flagged_ids = sample.loc[flagged, "id"].astype(str).tolist()
        raise RuntimeError(
            "The detector conflicts with the manual English baseline for IDs: "
            + "|".join(flagged_ids)
        )
    return sample.loc[:, LANGUAGE_VALIDATION_COLUMNS]


def remove_standalone_urls(text: str) -> str:
    """Remove standalone web-address tokens and preserve all other text."""

    return URL_PATTERN.sub("", text).strip()


def build_cleaned_corpus(data: pd.DataFrame) -> pd.DataFrame:
    """Keep relevant, language-eligible posts and prepare FinBERT text.

    Returns the final FinBERT-ready corpus and the larger relevant set with
    language diagnostics. Standalone URLs are removed only from ``finbert_text``;
    original title and selftext fields remain available for audit.
    """

    relevant = data.loc[data["relevant_candidate"]].copy()
    relevant = add_language_columns(relevant)
    relevant["finbert_text"] = relevant["full_text"].map(
        remove_standalone_urls
    )
    eligible = relevant.loc[
        relevant["language_status"].ne("detected_non_english")
    ].copy()
    eligible = eligible.sort_values(
        ["created_datetime_utc", "id"], kind="stable"
    ).reset_index(drop=True)
    return eligible.loc[:, CLEANED_COLUMNS], relevant


def validate_cleaned_corpus(
    cleaned: pd.DataFrame,
    raw: pd.DataFrame,
) -> None:
    """Validate identifiers, timestamps, source fields, and final text.

    Unique IDs prevent a Reddit post receiving more than one empirical weight,
    and source-field comparisons prove that cleaning did not rewrite evidence.
    """

    if cleaned.empty or cleaned["id"].duplicated().any():
        raise ValueError("The cleaned corpus is empty or contains duplicate IDs.")
    if set(cleaned["subreddit"].unique()) - set(SUBREDDITS):
        raise ValueError("The cleaned corpus contains an unexpected subreddit.")
    if not cleaned["date_utc"].between(START_DATE, END_DATE).all():
        raise ValueError("The cleaned corpus contains an out-of-period date.")
    timestamps = pd.to_datetime(
        cleaned["created_datetime_utc"], errors="coerce", utc=True
    )
    if timestamps.isna().any() or not timestamps.is_monotonic_increasing:
        raise ValueError("Cleaned UTC timestamps are invalid or unsorted.")
    if cleaned["title"].astype(str).str.strip().eq("").any():
        raise ValueError("A final post has a blank title.")
    if cleaned["full_text"].astype(str).str.strip().eq("").any():
        raise ValueError("A final post has blank full_text.")
    if cleaned["finbert_text"].astype(str).str.strip().eq("").any():
        raise ValueError("URL removal produced blank FinBERT text.")
    if not cleaned["relevant_candidate"].all():
        raise ValueError("The cleaned corpus contains an irrelevant post.")
    if cleaned["language_status"].eq("detected_non_english").any():
        raise ValueError("A confidently non-English post remains eligible.")

    source = raw.set_index("id").loc[cleaned["id"]]
    source.index = cleaned.index
    for column in ["title", "selftext"]:
        if not cleaned[column].equals(source[column]):
            raise ValueError(f"The cleaned corpus changed raw {column} text.")
    expected_full_text = np.where(
        cleaned["title_only"],
        cleaned["title"].astype(str),
        cleaned["title"].astype(str) + " " + cleaned["selftext"].astype(str),
    )
    if not np.array_equal(cleaned["full_text"], expected_full_text):
        raise ValueError("full_text does not follow the approved body rule.")


def add_summary_row(
    rows: list[dict[str, object]],
    section: str,
    metric: str,
    category: str,
    value: object,
) -> None:
    """Append one row to the readable long-form cleaning summary."""

    rows.append(
        {
            "section": section,
            "metric": metric,
            "category": category,
            "value": value,
        }
    )


def build_summary(
    raw: pd.DataFrame,
    relevant_with_language: pd.DataFrame,
    cleaned: pd.DataFrame,
    language_validation: pd.DataFrame,
    relevance_increase: int,
) -> pd.DataFrame:
    """Build the complete Phase 3A-to-Phase 3B transformation audit."""

    rows: list[dict[str, object]] = []
    non_english_removed = int(
        relevant_with_language["language_status"]
        .eq("detected_non_english")
        .sum()
    )
    pipeline = {
        "phase3a_raw_candidates": len(raw),
        "phase3b3_relevance_retained": EXPECTED_PHASE3B3_RETAINED,
        "final_relevance_retained": len(relevant_with_language),
        "increase_from_phase3b3": relevance_increase,
        "confidently_non_english_removed": non_english_removed,
        "final_finbert_ready_posts": len(cleaned),
    }
    for metric, value in pipeline.items():
        add_summary_row(rows, "pipeline", metric, "all", value)

    composition_groups = {
        "year": pd.to_datetime(cleaned["date_utc"]).dt.year,
        "subreddit": cleaned["subreddit"],
        "relevance_path": cleaned["relevance_path"],
        "body_status": cleaned["body_status"],
        "language_status": cleaned["language_status"],
    }
    for metric, values in composition_groups.items():
        counts = values.value_counts(dropna=False).sort_index()
        for category, count in counts.items():
            add_summary_row(rows, "final_composition", metric, str(category), count)

    sample_language_counts = (
        language_validation["language_status"]
        .value_counts(dropna=False)
        .sort_index()
    )
    for category, count in sample_language_counts.items():
        add_summary_row(
            rows,
            "language_validation_sample",
            "language_status",
            str(category),
            count,
        )

    quality = {
        "earliest_date": cleaned["date_utc"].min(),
        "latest_date": cleaned["date_utc"].max(),
        "unique_ids": cleaned["id"].nunique(),
        "duplicate_ids": int(cleaned["id"].duplicated().sum()),
        "blank_titles": int(cleaned["title"].astype(str).str.strip().eq("").sum()),
        "title_only_observations": int(cleaned["title_only"].sum()),
        "removed_body_retained": int(cleaned["body_status"].eq("removed").sum()),
        "deleted_body_retained": int(cleaned["body_status"].eq("deleted").sum()),
        "blank_body_retained": int(cleaned["body_status"].eq("blank").sum()),
        "out_of_period_dates": int(
            (~cleaned["date_utc"].between(START_DATE, END_DATE)).sum()
        ),
        "unexpected_subreddits": len(
            set(cleaned["subreddit"].unique()) - set(SUBREDDITS)
        ),
    }
    for metric, value in quality.items():
        add_summary_row(rows, "quality", metric, "all", value)
    return pd.DataFrame(rows, columns=["section", "metric", "category", "value"])


def write_outputs(
    cleaned: pd.DataFrame,
    summary: pd.DataFrame,
    language_validation: pd.DataFrame,
) -> None:
    """Write the final cleaned corpus and Phase 3B diagnostics."""

    CLEANED_REDDIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REDDIT_CLEANING_SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(CLEANED_REDDIT_FILE, index=False, encoding="utf-8-sig")
    summary.to_csv(
        REDDIT_CLEANING_SUMMARY_FILE, index=False, encoding="utf-8-sig"
    )
    language_validation.to_csv(
        REDDIT_LANGUAGE_VALIDATION_FILE, index=False, encoding="utf-8-sig"
    )


def print_diagnostics(
    relevant_with_language: pd.DataFrame,
    cleaned: pd.DataFrame,
    language_validation: pd.DataFrame,
    relevance_increase: int,
) -> None:
    """Print the requested final Phase 3B counts."""

    print(f"Phase 3B.3 retained: {EXPECTED_PHASE3B3_RETAINED:,}")
    print(f"Final relevance retained: {len(relevant_with_language):,}")
    print(f"Difference from Phase 3B.3: {relevance_increase:+,}")
    print(
        "Confidently non-English removed:",
        int(
            relevant_with_language["language_status"]
            .eq("detected_non_english")
            .sum()
        ),
    )
    print(f"Final FinBERT-ready posts: {len(cleaned):,}")
    print(
        "Posts by year:",
        cleaned.groupby(pd.to_datetime(cleaned["date_utc"]).dt.year)["id"]
        .nunique()
        .to_dict(),
    )
    print(
        "Posts by subreddit:",
        cleaned.groupby("subreddit")["id"].nunique().to_dict(),
    )
    print(
        "Posts by relevance path:",
        cleaned.groupby("relevance_path")["id"].nunique().to_dict(),
    )
    print(f"Title-only retained: {int(cleaned['title_only'].sum()):,}")
    print(
        "Unavailable bodies retained:",
        cleaned.loc[cleaned["title_only"]]
        .groupby("body_status")["id"]
        .nunique()
        .to_dict(),
    )
    print(
        "Language statuses:",
        cleaned.groupby("language_status")["id"].nunique().to_dict(),
    )
    print(
        "Manual-sample language statuses:",
        language_validation.groupby("language_status")["id"]
        .nunique()
        .to_dict(),
    )
    print(f"Earliest date: {cleaned['date_utc'].min()}")
    print(f"Latest date: {cleaned['date_utc'].max()}")


def main() -> None:
    """Finalize Phase 3B without running FinBERT or daily aggregation."""

    validate_configuration()
    DetectorFactory.seed = REDDIT_LANGUAGE_DETECTOR_SEED
    raw_hash_before = file_sha256(RAW_REDDIT_FILE)
    raw = load_raw_corpus()
    classified = apply_relevance_rules(add_usable_text(raw))
    relevance_increase = validate_relevance(classified)
    language_validation = validate_manual_language_sample()
    cleaned, relevant_with_language = build_cleaned_corpus(classified)
    validate_cleaned_corpus(cleaned, raw)
    summary = build_summary(
        raw,
        relevant_with_language,
        cleaned,
        language_validation,
        relevance_increase,
    )
    write_outputs(cleaned, summary, language_validation)
    if file_sha256(RAW_REDDIT_FILE) != raw_hash_before:
        raise RuntimeError("The immutable Phase 3A raw corpus changed.")
    print_diagnostics(
        relevant_with_language,
        cleaned,
        language_validation,
        relevance_increase,
    )
    print(f"Cleaned corpus SHA-256: {file_sha256(CLEANED_REDDIT_FILE)}")
    print(f"Cleaning summary SHA-256: {file_sha256(REDDIT_CLEANING_SUMMARY_FILE)}")
    print(f"Saved cleaned corpus to {CLEANED_REDDIT_FILE}")
    print(f"Saved cleaning summary to {REDDIT_CLEANING_SUMMARY_FILE}")
    print(f"Saved language validation to {REDDIT_LANGUAGE_VALIDATION_FILE}")


if __name__ == "__main__":
    main()
