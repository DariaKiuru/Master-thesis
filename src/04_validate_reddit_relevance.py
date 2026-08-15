"""Dry-run and validate the transparent Phase 3B Reddit relevance rule."""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# Allow ``python src/04_validate_reddit_relevance.py`` from the repository root.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from config import (  # noqa: E402
    END_DATE,
    RAW_REDDIT_FILE,
    REDDIT_CONFLICT_CONTEXT_TERMS,
    REDDIT_CRISIS_FINANCIAL_CONSEQUENCE_PHRASES,
    REDDIT_DIRECT_UKRAINE_CONTEXT_TERMS,
    REDDIT_EXTRACTION_KEYWORDS,
    REDDIT_FILTER_EXCLUDED_REVIEW_FILE,
    REDDIT_FILTER_INCLUDED_REVIEW_FILE,
    REDDIT_FILTER_REVIEW_RANDOM_SEED,
    REDDIT_FINANCIAL_CONTEXT_TERMS,
    REDDIT_RELEVANCE_FILTER_DRY_RUN_FILE,
    REDDIT_RUSSIA_CONTEXT_TERMS,
    START_DATE,
    SUBREDDITS,
)


REVIEW_SAMPLE_SIZE = 75
UNAVAILABLE_BODY_VALUES = {"", "[removed]", "[deleted]"}

DRY_RUN_COLUMNS = [
    "id",
    "date_utc",
    "subreddit",
    "title",
    "selftext",
    "extraction_matched_keywords_original",
    "matched_keywords_recomputed",
    "body_status",
    "title_only",
    "has_direct_ukraine_context",
    "has_russia_context",
    "has_conflict_context",
    "has_financial_context",
    "has_crisis_financial_consequence",
    "relevant_candidate",
    "relevance_path",
]
REVIEW_COLUMNS = [*DRY_RUN_COLUMNS, "review_group"]


def file_sha256(path: Path) -> str:
    """Return a checksum used to prove that the raw corpus remains unchanged."""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bounded_pattern(term: str) -> str:
    """Escape a term and allow flexible whitespace inside multi-word phrases."""

    return re.escape(term).replace(r"\ ", r"\s+")


def compile_term_pattern(terms: list[str]) -> re.Pattern[str]:
    """Compile case-insensitive phrase alternatives with non-word boundaries."""

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
CONFLICT_CONTEXT_PATTERN = compile_term_pattern(REDDIT_CONFLICT_CONTEXT_TERMS)
FINANCIAL_CONTEXT_PATTERN = compile_term_pattern(REDDIT_FINANCIAL_CONTEXT_TERMS)
CRISIS_FINANCIAL_PATTERN = compile_term_pattern(
    REDDIT_CRISIS_FINANCIAL_CONSEQUENCE_PHRASES
)
EXACT_RUSSIA_RUSSIAN_PATTERN = compile_term_pattern(["Russia", "Russian"])
SANCTIONS_INVASION_PATTERN = compile_term_pattern(
    [
        "sanction",
        "sanctions",
        "sanctioned",
        "sanctioning",
        "invasion",
        "invasions",
        "invade",
        "invades",
        "invaded",
        "invading",
    ]
)


def validate_configuration() -> None:
    """Check required vocabularies and the fixed thesis sample definition."""

    if SUBREDDITS != ["investing", "stocks", "StockMarket"]:
        raise ValueError("The Phase 3B subreddit universe is incorrect.")
    if START_DATE != "2021-01-01" or END_DATE != "2023-12-31":
        raise ValueError("The Phase 3B sample period is incorrect.")
    configured_groups = [
        REDDIT_DIRECT_UKRAINE_CONTEXT_TERMS,
        REDDIT_RUSSIA_CONTEXT_TERMS,
        REDDIT_CONFLICT_CONTEXT_TERMS,
        REDDIT_FINANCIAL_CONTEXT_TERMS,
        REDDIT_CRISIS_FINANCIAL_CONSEQUENCE_PHRASES,
    ]
    if any(not group for group in configured_groups):
        raise ValueError("Every relevance vocabulary must be non-empty.")
    if "gas" not in REDDIT_FINANCIAL_CONTEXT_TERMS:
        raise ValueError("The approved financial vocabulary must include gas.")


def load_raw_corpus() -> pd.DataFrame:
    """Load the immutable Phase 3A corpus without normalizing source text."""

    data = pd.read_csv(
        RAW_REDDIT_FILE,
        dtype={"id": "string", "subreddit": "string"},
        keep_default_na=False,
    )
    required = {
        "id",
        "date_utc",
        "subreddit",
        "title",
        "selftext",
        "matched_search_keywords",
    }
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Raw Reddit corpus lacks columns: {sorted(missing)}")
    if len(data) != 3_033 or data["id"].duplicated().any():
        raise ValueError("Expected exactly 3,033 unique Phase 3A Reddit posts.")
    if set(data["subreddit"].unique()) != set(SUBREDDITS):
        raise ValueError("Raw corpus contains an unexpected subreddit.")
    dates = pd.to_datetime(data["date_utc"], errors="coerce")
    if dates.isna().any() or not dates.between(
        START_DATE, END_DATE, inclusive="both"
    ).all():
        raise ValueError("Raw corpus contains an invalid or out-of-period date.")
    return data


def available_body_status(selftext: object) -> str:
    """Classify body availability without modifying the raw selftext value."""

    stripped = str(selftext).strip()
    lowered = stripped.casefold()
    if stripped == "":
        return "blank"
    if lowered == "[removed]":
        return "removed"
    if lowered == "[deleted]":
        return "deleted"
    return "available"


def recompute_extraction_keywords(text: str) -> str:
    """Recompute Phase 3A keyword provenance with bounded local matching."""

    matches = [
        keyword
        for keyword in REDDIT_EXTRACTION_KEYWORDS
        if EXTRACTION_PATTERNS[keyword].search(text)
    ]
    return "|".join(matches)


def assign_relevance_path(data: pd.DataFrame) -> pd.Series:
    """Label the single path, multiple paths, or exclusion for every post."""

    direct_path = (
        data["has_direct_ukraine_context"] & data["has_financial_context"]
    )
    conflict_path = (
        data["has_russia_context"]
        & data["has_conflict_context"]
        & data["has_financial_context"]
    )
    consequence_path = (
        data["has_russia_context"]
        & data["has_crisis_financial_consequence"]
    )
    path_count = (
        direct_path.astype(int)
        + conflict_path.astype(int)
        + consequence_path.astype(int)
    )
    return pd.Series(
        np.select(
            [
                path_count.gt(1),
                direct_path,
                conflict_path,
                consequence_path,
            ],
            [
                "multiple_paths",
                "direct_ukraine_financial",
                "russia_conflict_financial",
                "russia_crisis_financial_consequence",
            ],
            default="excluded",
        ),
        index=data.index,
        dtype="string",
    )


def apply_candidate_rule(raw: pd.DataFrame) -> pd.DataFrame:
    """Create usable text, bounded indicators, and the proposed relevance decision."""

    data = raw.copy()
    data["year"] = pd.to_datetime(data["date_utc"]).dt.year.astype(int)
    data["extraction_matched_keywords_original"] = data[
        "matched_search_keywords"
    ]
    data["body_status"] = data["selftext"].map(available_body_status)
    data["title_only"] = data["body_status"].ne("available")
    data["full_text"] = np.where(
        data["title_only"],
        data["title"].astype(str),
        data["title"].astype(str) + "\n" + data["selftext"].astype(str),
    )
    data["matched_keywords_recomputed"] = data["full_text"].map(
        recompute_extraction_keywords
    )
    data["has_direct_ukraine_context"] = data["full_text"].str.contains(
        DIRECT_UKRAINE_PATTERN, na=False
    )
    data["has_russia_context"] = data["full_text"].str.contains(
        RUSSIA_CONTEXT_PATTERN, na=False
    )
    data["has_conflict_context"] = data["full_text"].str.contains(
        CONFLICT_CONTEXT_PATTERN, na=False
    )
    data["has_financial_context"] = data["full_text"].str.contains(
        FINANCIAL_CONTEXT_PATTERN, na=False
    )
    data["has_crisis_financial_consequence"] = data["full_text"].str.contains(
        CRISIS_FINANCIAL_PATTERN, na=False
    )
    data["has_exact_russia_or_russian"] = data["full_text"].str.contains(
        EXACT_RUSSIA_RUSSIAN_PATTERN, na=False
    )
    data["has_sanctions_or_invasion"] = data["full_text"].str.contains(
        SANCTIONS_INVASION_PATTERN, na=False
    )
    data["relevance_path"] = assign_relevance_path(data)
    data["relevant_candidate"] = data["relevance_path"].ne("excluded")
    return data


def validate_rule(raw: pd.DataFrame, data: pd.DataFrame) -> None:
    """Validate provenance correction, usable text, and exact Boolean logic."""

    if len(data) != len(raw) or not data["id"].equals(raw["id"]):
        raise ValueError("The dry run changed the raw corpus membership or order.")
    if not data["title"].equals(raw["title"]) or not data["selftext"].equals(
        raw["selftext"]
    ):
        raise ValueError("The dry run changed raw Reddit title or selftext.")

    if recompute_extraction_keywords("Russian stocks") != "Russian":
        raise ValueError("Bounded provenance still confuses Russian with Russia.")
    if recompute_extraction_keywords("Russia stocks") != "Russia":
        raise ValueError("Bounded provenance does not identify Russia correctly.")
    if FINANCIAL_CONTEXT_PATTERN.search("vegas"):
        raise ValueError("The bounded gas term incorrectly matches vegas.")

    expected_relevant = (
        data["has_direct_ukraine_context"] & data["has_financial_context"]
    ) | (
        data["has_russia_context"]
        & data["has_conflict_context"]
        & data["has_financial_context"]
    ) | (
        data["has_russia_context"]
        & data["has_crisis_financial_consequence"]
    )
    if not np.array_equal(
        data["relevant_candidate"].to_numpy(dtype=bool),
        expected_relevant.to_numpy(dtype=bool),
    ):
        raise ValueError("relevant_candidate does not implement the approved formula.")
    if not data.loc[data["title_only"], "full_text"].equals(
        data.loc[data["title_only"], "title"].astype(str)
    ):
        raise ValueError("A title-only post used unavailable body text.")
    if not data.loc[~data["title_only"], "full_text"].equals(
        data.loc[~data["title_only"], "title"].astype(str)
        + "\n"
        + data.loc[~data["title_only"], "selftext"].astype(str)
    ):
        raise ValueError("An available body was omitted from usable text.")
    allowed_paths = {
        "direct_ukraine_financial",
        "russia_conflict_financial",
        "russia_crisis_financial_consequence",
        "multiple_paths",
        "excluded",
    }
    if set(data["relevance_path"].unique()) - allowed_paths:
        raise ValueError("An unexpected relevance_path was produced.")


def balanced_take(
    data: pd.DataFrame,
    eligible: pd.Series,
    selected_ids: set[str],
    requested_count: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Take up to a quota while cycling across year/subreddit strata."""

    pool = data.loc[eligible & ~data["id"].isin(selected_ids)].copy()
    count = min(requested_count, len(pool))
    if count == 0:
        return pool
    queues: dict[tuple[int, str], list[int]] = {}
    for stratum, group in pool.groupby(["year", "subreddit"], sort=True):
        queues[stratum] = rng.permutation(group.index.to_numpy()).tolist()
    strata = list(queues)
    rng.shuffle(strata)
    chosen: list[int] = []
    while len(chosen) < count:
        for stratum in strata:
            if queues[stratum] and len(chosen) < count:
                chosen.append(queues[stratum].pop())
    return data.loc[chosen].copy()


def build_review_sample(data: pd.DataFrame, included: bool) -> pd.DataFrame:
    """Create one deterministic, deliberately varied filter-validation sample."""

    rng = np.random.default_rng(
        REDDIT_FILTER_REVIEW_RANDOM_SEED + (0 if included else 1)
    )
    population = data["relevant_candidate"].eq(included)
    selected_ids: set[str] = set()
    selections: list[pd.DataFrame] = []

    def add_group(label: str, condition: pd.Series, count: int) -> None:
        chosen = balanced_take(
            data, population & condition, selected_ids, count, rng
        )
        if chosen.empty:
            return
        chosen["review_group"] = label
        selected_ids.update(chosen["id"].astype(str))
        selections.append(chosen)

    if included:
        for path in [
            "direct_ukraine_financial",
            "russia_conflict_financial",
            "russia_crisis_financial_consequence",
            "multiple_paths",
        ]:
            add_group(f"path_{path}", data["relevance_path"].eq(path), 12)
        add_group("title_only", data["title_only"], 12)
        add_group("sanctions_or_invasion", data["has_sanctions_or_invasion"], 8)
        add_group("Russia_or_Russian", data["has_exact_russia_or_russian"], 5)
    else:
        add_group("title_only", data["title_only"], 15)
        add_group("Russia_or_Russian", data["has_exact_russia_or_russian"], 20)
        add_group(
            "direct_ukraine_but_excluded",
            data["has_direct_ukraine_context"],
            10,
        )
        add_group(
            "sanctions_or_invasion_but_excluded",
            data["has_sanctions_or_invasion"],
            15,
        )

    selected_count = sum(len(frame) for frame in selections)
    add_group("general", pd.Series(True, index=data.index), REVIEW_SAMPLE_SIZE - selected_count)
    sample = pd.concat(selections, ignore_index=True)
    validate_review_sample(sample, data, included)
    return sample.loc[:, REVIEW_COLUMNS].sort_values(
        ["review_group", "date_utc", "subreddit", "id"], kind="stable"
    ).reset_index(drop=True)


def validate_review_sample(
    sample: pd.DataFrame,
    full_data: pd.DataFrame,
    included: bool,
) -> None:
    """Confirm each validation sample covers the requested dimensions."""

    label = "included" if included else "excluded"
    if len(sample) != REVIEW_SAMPLE_SIZE or sample["id"].duplicated().any():
        raise ValueError(f"The {label} review sample must contain 75 unique posts.")
    if not sample["relevant_candidate"].eq(included).all():
        raise ValueError(f"The {label} review sample contains the wrong decision.")
    if set(sample["year"]) != {2021, 2022, 2023}:
        raise ValueError(f"The {label} review sample lacks a thesis year.")
    if set(sample["subreddit"]) != set(SUBREDDITS):
        raise ValueError(f"The {label} review sample lacks a subreddit.")
    required_masks = {
        "title-only": sample["title_only"],
        "Russia/Russian": sample["has_exact_russia_or_russian"],
        "direct Ukraine": sample["has_direct_ukraine_context"],
        "sanctions/invasion": sample["has_sanctions_or_invasion"],
    }
    for name, mask in required_masks.items():
        if not mask.any():
            raise ValueError(f"The {label} review sample lacks {name} posts.")
    if included:
        available_paths = set(
            full_data.loc[
                full_data["relevant_candidate"], "relevance_path"
            ].unique()
        )
        if set(sample["relevance_path"]) != available_paths:
            raise ValueError("Included review sample lacks a relevance path.")


def output_dry_run(data: pd.DataFrame) -> pd.DataFrame:
    """Return the auditable public dry-run schema."""

    dry_run = data.loc[:, DRY_RUN_COLUMNS].copy()
    if len(dry_run) != 3_033 or dry_run["id"].duplicated().any():
        raise ValueError("Dry-run output must retain all 3,033 unique raw posts.")
    return dry_run


def write_outputs(
    dry_run: pd.DataFrame,
    included_review: pd.DataFrame,
    excluded_review: pd.DataFrame,
) -> None:
    """Write only diagnostics; never create the final cleaned Reddit dataset."""

    REDDIT_RELEVANCE_FILTER_DRY_RUN_FILE.parent.mkdir(parents=True, exist_ok=True)
    dry_run.to_csv(
        REDDIT_RELEVANCE_FILTER_DRY_RUN_FILE, index=False, encoding="utf-8-sig"
    )
    included_review.to_csv(
        REDDIT_FILTER_INCLUDED_REVIEW_FILE, index=False, encoding="utf-8-sig"
    )
    excluded_review.to_csv(
        REDDIT_FILTER_EXCLUDED_REVIEW_FILE, index=False, encoding="utf-8-sig"
    )


def print_diagnostics(data: pd.DataFrame) -> None:
    """Print the core dry-run counts needed for manual validation."""

    retained = data.loc[data["relevant_candidate"]]
    excluded = data.loc[~data["relevant_candidate"]]
    keyword_sets = data["matched_keywords_recomputed"].str.split("|").map(
        lambda terms: {term for term in terms if term}
    )
    russia = keyword_sets.map(lambda terms: "Russia" in terms)
    russian = keyword_sets.map(lambda terms: "Russian" in terms)
    print(f"Raw candidate posts: {len(data):,}")
    print(f"Retained by proposed rule: {len(retained):,}")
    print(f"Excluded by proposed rule: {len(excluded):,}")
    print(f"Retention percentage: {len(retained) / len(data):.2%}")
    print("Retained by year:", retained.groupby("year")["id"].nunique().to_dict())
    print(
        "Retained by subreddit:",
        retained.groupby("subreddit")["id"].nunique().to_dict(),
    )
    print(
        "Retained by relevance path:",
        retained.groupby("relevance_path")["id"].nunique().to_dict(),
    )
    print(f"Title-only retained: {int(retained['title_only'].sum()):,}")
    print(
        "Retained body states:",
        retained.groupby("body_status")["id"].nunique().to_dict(),
    )
    print(
        "Corrected keyword counts:",
        {
            "Russia": int(russia.sum()),
            "Russian": int(russian.sum()),
            "Russia_only": int((russia & ~russian).sum()),
            "Russian_only": int((russian & ~russia).sum()),
            "both": int((russia & russian).sum()),
            "no_recomputed_keyword": int(keyword_sets.map(len).eq(0).sum()),
        },
    )
    print(
        "Retained without direct Ukraine context:",
        int((~retained["has_direct_ukraine_context"]).sum()),
    )
    print(
        "Excluded containing exact Russia or Russian:",
        int(excluded["has_exact_russia_or_russian"].sum()),
    )


def main() -> None:
    """Run Phase 3B.2 validation without finalizing or scoring posts."""

    validate_configuration()
    raw_hash_before = file_sha256(RAW_REDDIT_FILE)
    raw = load_raw_corpus()
    data = apply_candidate_rule(raw)
    validate_rule(raw, data)
    dry_run = output_dry_run(data)
    included_review = build_review_sample(data, included=True)
    excluded_review = build_review_sample(data, included=False)
    write_outputs(dry_run, included_review, excluded_review)
    if file_sha256(RAW_REDDIT_FILE) != raw_hash_before:
        raise RuntimeError("The immutable Phase 3A raw corpus changed.")
    print_diagnostics(data)
    print(f"Saved dry run to {REDDIT_RELEVANCE_FILTER_DRY_RUN_FILE}")
    print(f"Saved included review to {REDDIT_FILTER_INCLUDED_REVIEW_FILE}")
    print(f"Saved excluded review to {REDDIT_FILTER_EXCLUDED_REVIEW_FILE}")


if __name__ == "__main__":
    main()
