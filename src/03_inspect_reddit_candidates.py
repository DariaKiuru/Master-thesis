"""Profile the Phase 3A Reddit corpus and create a manual-review sample."""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# Allow ``python src/03_inspect_reddit_candidates.py`` from the repository root.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from config import (  # noqa: E402
    END_DATE,
    RAW_REDDIT_FILE,
    REDDIT_CANDIDATE_CORPUS_PROFILE_FILE,
    REDDIT_EXTRACTION_KEYWORDS,
    REDDIT_RELEVANCE_REVIEW_SAMPLE_FILE,
    REDDIT_REVIEW_RANDOM_SEED,
    START_DATE,
    SUBREDDITS,
)


REVIEW_SAMPLE_SIZE = 150
REVIEW_OUTPUT_COLUMNS = [
    "id",
    "date_utc",
    "subreddit",
    "title",
    "selftext",
    "matched_search_keywords",
    "review_group",
]
REQUIRED_RAW_COLUMNS = set(REVIEW_OUTPUT_COLUMNS) - {"review_group"}

DIRECT_UKRAINE_TERMS = {"Ukraine", "Ukrainian", "Kyiv", "Kiev"}
DIAGNOSTIC_CONTEXT_TERMS = [
    "war",
    "invasion",
    "troops",
    "military",
    "conflict",
    "sanctions",
    "NATO",
    "Crimea",
    "Donbas",
    "Donetsk",
    "Luhansk",
    "Ukraine",
    "Ukrainian",
    "Kyiv",
    "Kiev",
]
CONTEXT_PATTERNS = {
    term: re.compile(rf"(?<!\w){re.escape(term)}(?!\w)", re.IGNORECASE)
    for term in DIAGNOSTIC_CONTEXT_TERMS
}


def file_sha256(path: Path) -> str:
    """Return a source-file checksum so the inspection cannot alter it silently."""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_candidate_corpus() -> pd.DataFrame:
    """Read and validate the immutable Phase 3A candidate corpus."""

    if not RAW_REDDIT_FILE.exists():
        raise FileNotFoundError(f"Raw Reddit corpus not found: {RAW_REDDIT_FILE}")
    data = pd.read_csv(
        RAW_REDDIT_FILE,
        dtype={"id": "string", "subreddit": "string"},
        keep_default_na=False,
    )
    missing = REQUIRED_RAW_COLUMNS - set(data.columns)
    if missing:
        raise ValueError(f"Raw Reddit corpus lacks columns: {sorted(missing)}")
    if len(data) != 3_033:
        raise ValueError(f"Expected 3,033 candidate posts, found {len(data):,}.")
    if data["id"].duplicated().any() or data["id"].astype(str).str.strip().eq("").any():
        raise ValueError("Candidate Reddit IDs must be present and globally unique.")
    if set(data["subreddit"].unique()) != set(SUBREDDITS):
        raise ValueError("Candidate corpus does not contain exactly the approved subreddits.")

    parsed_dates = pd.to_datetime(data["date_utc"], errors="coerce")
    if parsed_dates.isna().any():
        raise ValueError("Candidate corpus contains an invalid date_utc value.")
    if not parsed_dates.between(START_DATE, END_DATE, inclusive="both").all():
        raise ValueError("Candidate corpus contains dates outside the thesis period.")
    if data["matched_search_keywords"].astype(str).str.strip().eq("").any():
        raise ValueError("Candidate corpus contains empty keyword provenance.")

    data = data.copy()
    data["year"] = parsed_dates.dt.year.astype(int)
    data["keyword_list"] = data["matched_search_keywords"].str.split("|")
    data["keyword_set"] = data["keyword_list"].map(set)
    data["matched_keyword_count"] = data["keyword_list"].str.len().astype(int)
    data["selftext_stripped"] = data["selftext"].astype(str).str.strip()
    data["title_stripped"] = data["title"].astype(str).str.strip()
    data["combined_text"] = (
        data["title"].astype(str) + "\n" + data["selftext"].astype(str)
    )
    for term, pattern in CONTEXT_PATTERNS.items():
        data[f"context_{term}"] = data["combined_text"].str.contains(
            pattern, na=False
        )
    context_columns = [f"context_{term}" for term in DIAGNOSTIC_CONTEXT_TERMS]
    data["has_diagnostic_context"] = data[context_columns].any(axis=1)
    return data


def add_profile_row(
    rows: list[dict[str, object]],
    total_posts: int,
    metric: str,
    count: int,
    definition: str,
    *,
    year: int | str = "",
    subreddit: str = "",
    category: str = "",
) -> None:
    """Append one long-form, auditable profile statistic."""

    rows.append(
        {
            "metric": metric,
            "year": year,
            "subreddit": subreddit,
            "category": category,
            "count": int(count),
            "proportion_of_corpus": round(count / total_posts, 6),
            "definition": definition,
        }
    )


def build_corpus_profile(data: pd.DataFrame) -> pd.DataFrame:
    """Build descriptive corpus counts without making inclusion decisions."""

    total = len(data)
    rows: list[dict[str, object]] = []
    add_profile_row(
        rows, total, "total_candidate_posts", total,
        "All unique Phase 3A candidate posts.", category="all",
    )
    for year in range(2021, 2024):
        count = int(data["year"].eq(year).sum())
        add_profile_row(
            rows, total, "posts_by_year", count,
            "Unique candidate posts by UTC calendar year.", year=year,
        )
    for subreddit in SUBREDDITS:
        count = int(data["subreddit"].eq(subreddit).sum())
        add_profile_row(
            rows, total, "posts_by_subreddit", count,
            "Unique candidate posts by source subreddit.", subreddit=subreddit,
        )
    for year in range(2021, 2024):
        for subreddit in SUBREDDITS:
            count = int(
                (data["year"].eq(year) & data["subreddit"].eq(subreddit)).sum()
            )
            add_profile_row(
                rows, total, "posts_by_year_and_subreddit", count,
                "Unique candidate posts by UTC year and source subreddit.",
                year=year, subreddit=subreddit,
            )
    for keyword_count in sorted(data["matched_keyword_count"].unique()):
        count = int(data["matched_keyword_count"].eq(keyword_count).sum())
        add_profile_row(
            rows, total, "matched_keyword_count_distribution", count,
            "Number of locally matched extraction keywords recorded for a post.",
            category=str(keyword_count),
        )

    only_russia = data["keyword_set"].map(lambda terms: terms == {"Russia"})
    only_russian = data["keyword_set"].map(lambda terms: terms == {"Russian"})
    has_russia_term = data["keyword_set"].map(
        lambda terms: bool(terms & {"Russia", "Russian"})
    )
    lacks_direct_ukraine = data["keyword_set"].map(
        lambda terms: not bool(terms & DIRECT_UKRAINE_TERMS)
    )
    add_profile_row(
        rows, total, "posts_matching_only_russia", int(only_russia.sum()),
        "matched_search_keywords contains Russia and no other extraction keyword.",
        category="Russia_only",
    )
    add_profile_row(
        rows, total, "posts_matching_only_russian", int(only_russian.sum()),
        "matched_search_keywords contains Russian and no other extraction keyword.",
        category="Russian_only",
    )
    add_profile_row(
        rows,
        total,
        "russia_or_russian_without_direct_ukraine_terms",
        int((has_russia_term & lacks_direct_ukraine).sum()),
        "Russia or Russian is matched, while Ukraine, Ukrainian, Kyiv, and Kiev are not.",
        category="ambiguous_russia_russian",
    )

    blank_body = data["selftext_stripped"].eq("")
    removed_body = data["selftext_stripped"].eq("[removed]")
    deleted_body = data["selftext_stripped"].eq("[deleted]")
    unavailable_body = blank_body | removed_body | deleted_body
    for category, mask in [
        ("blank", blank_body),
        ("removed", removed_body),
        ("deleted", deleted_body),
    ]:
        add_profile_row(
            rows, total, "selftext_status", int(mask.sum()),
            "Raw selftext availability state; no posts are removed by this diagnostic.",
            category=category,
        )
    add_profile_row(
        rows,
        total,
        "usable_title_with_unavailable_body",
        int((data["title_stripped"].ne("") & unavailable_body).sum()),
        "Non-blank title with blank, [removed], or [deleted] selftext.",
        category="title_available_body_unavailable",
    )
    add_profile_row(
        rows,
        total,
        "diagnostic_crisis_context_any",
        int(data["has_diagnostic_context"].sum()),
        "Title + selftext contains at least one diagnostic crisis-context term; not a filter.",
        category="any_context_term",
    )
    for term in DIAGNOSTIC_CONTEXT_TERMS:
        add_profile_row(
            rows,
            total,
            "diagnostic_context_term",
            int(data[f"context_{term}"].sum()),
            "Case-insensitive whole-term match in title + selftext; diagnostic only.",
            category=term,
        )
    return pd.DataFrame(rows)


def balanced_take(
    data: pd.DataFrame,
    eligible: pd.Series,
    selected_ids: set[str],
    count: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Select deterministically while cycling across available year/subreddit strata."""

    pool = data.loc[eligible & ~data["id"].isin(selected_ids)].copy()
    if len(pool) < count:
        raise ValueError(f"Review stratum needs {count} rows but only {len(pool)} remain.")

    queues: dict[tuple[int, str], list[int]] = {}
    for stratum, group in pool.groupby(["year", "subreddit"], sort=True):
        queues[stratum] = rng.permutation(group.index.to_numpy()).tolist()
    strata = list(queues)
    rng.shuffle(strata)
    chosen: list[int] = []
    while len(chosen) < count:
        progressed = False
        for stratum in strata:
            if queues[stratum] and len(chosen) < count:
                chosen.append(queues[stratum].pop())
                progressed = True
        if not progressed:
            raise RuntimeError("Balanced review sampling exhausted unexpectedly.")
    return data.loc[chosen].copy()


def build_review_sample(data: pd.DataFrame) -> pd.DataFrame:
    """Create a fixed-seed sample spanning high-value manual-review categories."""

    rng = np.random.default_rng(REDDIT_REVIEW_RANDOM_SEED)
    selected_ids: set[str] = set()
    selections: list[pd.DataFrame] = []

    def add_group(label: str, eligible: pd.Series, count: int) -> None:
        chosen = balanced_take(data, eligible, selected_ids, count, rng)
        chosen["review_group"] = label
        selected_ids.update(chosen["id"].astype(str))
        selections.append(chosen)

    # Guarantee representation of every extraction keyword, including rare terms.
    for keyword in REDDIT_EXTRACTION_KEYWORDS:
        if keyword == "Russia":
            eligible = data["keyword_set"].map(lambda terms: terms == {"Russia"})
        elif keyword == "Russian":
            eligible = data["keyword_set"].map(
                lambda terms: "Russian" in terms
                and terms.issubset({"Russia", "Russian"})
            )
        else:
            eligible = data["keyword_set"].map(lambda terms: keyword in terms)
        add_group(f"keyword_coverage_{keyword}", eligible, 1)

    stripped = data["selftext_stripped"]
    add_group("removed_body", stripped.eq("[removed]"), 12)
    add_group("deleted_body", stripped.eq("[deleted]"), 8)
    add_group("blank_body", stripped.eq(""), 10)
    add_group(
        "Russia_only",
        data["keyword_set"].map(lambda terms: terms == {"Russia"}),
        24,
    )
    add_group(
        "Russia_Russian_only",
        data["keyword_set"].map(
            lambda terms: terms == {"Russia", "Russian"}
        ),
        20,
    )
    add_group(
        "Ukraine_direct",
        data["keyword_set"].map(lambda terms: bool(terms & DIRECT_UKRAINE_TERMS)),
        20,
    )
    add_group(
        "sanctions",
        data["keyword_set"].map(lambda terms: "sanctions" in terms),
        10,
    )
    add_group(
        "invasion",
        data["keyword_set"].map(lambda terms: "invasion" in terms),
        10,
    )
    add_group("multi_keyword", data["matched_keyword_count"].gt(1), 20)
    add_group("other_keyword", pd.Series(True, index=data.index), 5)

    sample = pd.concat(selections, ignore_index=True)
    sample = sample.loc[:, REVIEW_OUTPUT_COLUMNS].sort_values(
        ["review_group", "date_utc", "subreddit", "id"], kind="stable"
    ).reset_index(drop=True)
    validate_review_sample(sample)
    return sample


def validate_review_sample(sample: pd.DataFrame) -> None:
    """Confirm sample size, provenance coverage, and raw text preservation."""

    if len(sample) != REVIEW_SAMPLE_SIZE or sample["id"].duplicated().any():
        raise ValueError("Manual-review sample must contain 150 unique posts.")
    if list(sample.columns) != REVIEW_OUTPUT_COLUMNS:
        raise ValueError("Manual-review sample schema is incorrect.")
    years = set(pd.to_datetime(sample["date_utc"]).dt.year)
    if years != {2021, 2022, 2023}:
        raise ValueError("Manual-review sample does not cover all three years.")
    if set(sample["subreddit"]) != set(SUBREDDITS):
        raise ValueError("Manual-review sample does not cover all three subreddits.")
    keyword_union: set[str] = set()
    for value in sample["matched_search_keywords"]:
        keyword_union.update(str(value).split("|"))
    if keyword_union != set(REDDIT_EXTRACTION_KEYWORDS):
        raise ValueError("Manual-review sample does not cover every extraction keyword.")
    keyword_counts = sample["matched_search_keywords"].str.count(r"\|") + 1
    if not keyword_counts.eq(1).any() or not keyword_counts.gt(1).any():
        raise ValueError("Sample must include single- and multi-keyword posts.")
    if sample["review_group"].astype(str).str.strip().eq("").any():
        raise ValueError("Every sampled post must have a review_group.")


def validate_profile(profile: pd.DataFrame, total_posts: int) -> None:
    """Check profile schema and core count reconciliations."""

    expected_columns = [
        "metric",
        "year",
        "subreddit",
        "category",
        "count",
        "proportion_of_corpus",
        "definition",
    ]
    if list(profile.columns) != expected_columns:
        raise ValueError("Candidate-corpus profile schema is incorrect.")
    total_row = profile.loc[profile["metric"].eq("total_candidate_posts"), "count"]
    if len(total_row) != 1 or int(total_row.iloc[0]) != total_posts:
        raise ValueError("Profile total does not reconcile to the candidate corpus.")
    for metric in ["posts_by_year", "posts_by_subreddit"]:
        if int(profile.loc[profile["metric"].eq(metric), "count"].sum()) != total_posts:
            raise ValueError(f"Profile metric {metric} does not reconcile to total posts.")


def write_outputs(sample: pd.DataFrame, profile: pd.DataFrame) -> None:
    """Write only Phase 3B.1 diagnostics, never the raw input corpus."""

    REDDIT_RELEVANCE_REVIEW_SAMPLE_FILE.parent.mkdir(parents=True, exist_ok=True)
    REDDIT_CANDIDATE_CORPUS_PROFILE_FILE.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(REDDIT_RELEVANCE_REVIEW_SAMPLE_FILE, index=False, encoding="utf-8-sig")
    profile.to_csv(REDDIT_CANDIDATE_CORPUS_PROFILE_FILE, index=False)


def main() -> None:
    """Generate deterministic diagnostics without filtering candidate posts."""

    source_hash_before = file_sha256(RAW_REDDIT_FILE)
    data = load_candidate_corpus()
    sample = build_review_sample(data)
    profile = build_corpus_profile(data)
    validate_profile(profile, len(data))
    write_outputs(sample, profile)
    source_hash_after = file_sha256(RAW_REDDIT_FILE)
    if source_hash_before != source_hash_after:
        raise RuntimeError("The immutable raw Reddit corpus changed during inspection.")
    print(f"Profiled {len(data):,} candidate posts without filtering.")
    print(f"Saved {len(sample)} review rows to {REDDIT_RELEVANCE_REVIEW_SAMPLE_FILE}")
    print(f"Saved corpus profile to {REDDIT_CANDIDATE_CORPUS_PROFILE_FILE}")


if __name__ == "__main__":
    main()
