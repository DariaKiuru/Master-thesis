"""Compare the Phase 3B.2 Reddit relevance rule with its Phase 3B.3 refinement."""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# Allow ``python src/05_compare_reddit_relevance_rules.py`` from the repo root.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from config import (  # noqa: E402
    END_DATE,
    RAW_REDDIT_FILE,
    REDDIT_DIRECT_UKRAINE_CONTEXT_TERMS,
    REDDIT_EXTRACTION_KEYWORDS,
    REDDIT_REFINED_CONFLICT_CONTEXT_TERMS,
    REDDIT_REFINED_CRISIS_FINANCIAL_CONSEQUENCE_PHRASES,
    REDDIT_REFINED_FINANCIAL_CONTEXT_TERMS,
    REDDIT_RELEVANCE_FILTER_DRY_RUN_FILE,
    REDDIT_RELEVANCE_RULE_COMPARISON_FILE,
    REDDIT_RULE_COMPARISON_RANDOM_SEED,
    REDDIT_RULE_NEWLY_EXCLUDED_REVIEW_FILE,
    REDDIT_RULE_NEWLY_INCLUDED_REVIEW_FILE,
    REDDIT_RUSSIA_CONTEXT_TERMS,
    START_DATE,
    SUBREDDITS,
)


EXPECTED_POSTS = 3_033
EXPECTED_OLD_RETAINED = 1_669
MAX_CHANGED_REVIEW_ROWS = 100

COMPARISON_COLUMNS = [
    "id",
    "date_utc",
    "subreddit",
    "title",
    "matched_keywords_recomputed",
    "body_status",
    "title_only",
    "has_direct_ukraine_context",
    "has_russia_context",
    "old_has_conflict_context",
    "has_conflict_context",
    "old_has_financial_context",
    "has_financial_context",
    "old_has_crisis_financial_consequence",
    "has_crisis_financial_consequence",
    "geopolitical_crisis_financial",
    "crisis_specific_financial_consequence",
    "old_relevant_candidate",
    "new_relevant_candidate",
    "old_relevance_path",
    "new_relevance_path",
    "decision_changed",
]

REVIEW_COLUMNS = [
    *COMPARISON_COLUMNS,
    "selftext",
    "change_type",
]

KNOWN_CASES = {
    "sxi55o": "hedge/make money from Russian invasion",
    "tcr6vc": "Goldman Sachs profits from war in Ukraine",
    "mi5ox3": "Russia/Ukraine tensions and money",
    "sagrpp": "defense stocks and Russia/Eastern Europe tensions",
    "ld2r3o": "generic Ukrainian mining investment",
    "lgeo3w": "generic Ukrainian agriculture investment",
    "m2bubi": "company discussion with incidental Ukraine reference",
    "onb2s6": "unrelated market wording with Ukraine reference",
    "mf8cmo": "generic pre-war Russian bonds",
    "mf8gqb": "generic Russian government bonds",
    "n9u9pp": "generic Ukrainian property investment",
    "li0rpg": "long company DD with incidental geopolitics",
}


def file_sha256(path: Path) -> str:
    """Return a checksum proving that the Phase 3A corpus was not modified."""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bounded_pattern(term: str) -> str:
    """Escape a term while allowing flexible whitespace in phrases."""

    return re.escape(term).replace(r"\ ", r"\s+")


def compile_term_pattern(terms: list[str]) -> re.Pattern[str]:
    """Compile case-insensitive alternatives with proper word boundaries."""

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
REFINED_CONFLICT_PATTERN = compile_term_pattern(
    REDDIT_REFINED_CONFLICT_CONTEXT_TERMS
)
REFINED_FINANCIAL_PATTERN = compile_term_pattern(
    REDDIT_REFINED_FINANCIAL_CONTEXT_TERMS
)
REFINED_CONSEQUENCE_PATTERN = compile_term_pattern(
    REDDIT_REFINED_CRISIS_FINANCIAL_CONSEQUENCE_PHRASES
)


def parse_boolean(series: pd.Series, column_name: str) -> pd.Series:
    """Read an audited Boolean CSV column without truthy-string ambiguity."""

    normalized = series.astype(str).str.strip().str.casefold()
    unexpected = set(normalized.unique()) - {"true", "false"}
    if unexpected:
        raise ValueError(
            f"Unexpected values in {column_name}: {sorted(unexpected)}"
        )
    return normalized.eq("true")


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the immutable corpus and the fixed Phase 3B.2 baseline."""

    raw = pd.read_csv(
        RAW_REDDIT_FILE,
        dtype={"id": "string", "subreddit": "string"},
        keep_default_na=False,
    )
    old = pd.read_csv(
        REDDIT_RELEVANCE_FILTER_DRY_RUN_FILE,
        dtype={"id": "string", "subreddit": "string"},
        keep_default_na=False,
    )
    required_raw = {
        "id",
        "date_utc",
        "subreddit",
        "title",
        "selftext",
    }
    required_old = {
        "id",
        "matched_keywords_recomputed",
        "has_conflict_context",
        "has_financial_context",
        "has_crisis_financial_consequence",
        "relevant_candidate",
        "relevance_path",
    }
    if missing := required_raw - set(raw.columns):
        raise ValueError(f"Raw corpus lacks columns: {sorted(missing)}")
    if missing := required_old - set(old.columns):
        raise ValueError(f"Phase 3B.2 dry run lacks columns: {sorted(missing)}")
    for label, data in [("raw", raw), ("Phase 3B.2", old)]:
        if len(data) != EXPECTED_POSTS or data["id"].duplicated().any():
            raise ValueError(f"Expected 3,033 unique posts in {label} data.")
    if set(raw["id"]) != set(old["id"]):
        raise ValueError("Phase 3B.2 and raw corpus post IDs differ.")
    if set(raw["subreddit"].unique()) != set(SUBREDDITS):
        raise ValueError("Raw corpus contains an unexpected subreddit.")
    dates = pd.to_datetime(raw["date_utc"], errors="coerce")
    if dates.isna().any() or not dates.between(
        START_DATE, END_DATE, inclusive="both"
    ).all():
        raise ValueError("Raw corpus contains an invalid or out-of-period date.")
    return raw, old


def body_status(selftext: object) -> str:
    """Classify unavailable bodies without changing their raw representation."""

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
    """Reproduce corrected Phase 3B.2 provenance with bounded matching."""

    return "|".join(
        keyword
        for keyword in REDDIT_EXTRACTION_KEYWORDS
        if EXTRACTION_PATTERNS[keyword].search(text)
    )


def assign_new_path(data: pd.DataFrame) -> pd.Series:
    """Assign the refined rule's two transparent paths."""

    path_count = (
        data["geopolitical_crisis_financial"].astype(int)
        + data["crisis_specific_financial_consequence"].astype(int)
    )
    return pd.Series(
        np.select(
            [
                path_count.eq(2),
                data["geopolitical_crisis_financial"],
                data["crisis_specific_financial_consequence"],
            ],
            [
                "multiple_paths",
                "geopolitical_crisis_financial",
                "crisis_specific_financial_consequence",
            ],
            default="excluded",
        ),
        index=data.index,
        dtype="string",
    )


def build_comparison(raw: pd.DataFrame, old: pd.DataFrame) -> pd.DataFrame:
    """Apply the refined rule and retain the fixed old decision beside it."""

    old_fields = old.loc[
        :,
        [
            "id",
            "matched_keywords_recomputed",
            "has_conflict_context",
            "has_financial_context",
            "has_crisis_financial_consequence",
            "relevant_candidate",
            "relevance_path",
        ],
    ].rename(
        columns={
            "matched_keywords_recomputed": "old_matched_keywords_recomputed",
            "has_conflict_context": "old_has_conflict_context",
            "has_financial_context": "old_has_financial_context",
            "has_crisis_financial_consequence": (
                "old_has_crisis_financial_consequence"
            ),
            "relevant_candidate": "old_relevant_candidate",
            "relevance_path": "old_relevance_path",
        }
    )
    data = raw.merge(old_fields, on="id", how="left", validate="one_to_one")
    for column in [
        "old_has_conflict_context",
        "old_has_financial_context",
        "old_has_crisis_financial_consequence",
        "old_relevant_candidate",
    ]:
        data[column] = parse_boolean(data[column], column)

    data["year"] = pd.to_datetime(data["date_utc"]).dt.year.astype(int)
    data["body_status"] = data["selftext"].map(body_status)
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
        REFINED_CONFLICT_PATTERN, na=False
    )
    data["has_financial_context"] = data["full_text"].str.contains(
        REFINED_FINANCIAL_PATTERN, na=False
    )
    data["has_crisis_financial_consequence"] = data[
        "full_text"
    ].str.contains(REFINED_CONSEQUENCE_PATTERN, na=False)

    geopolitical = (
        data["has_direct_ukraine_context"] | data["has_russia_context"]
    ) & data["has_conflict_context"] & data["has_financial_context"]
    consequence = (
        data["has_direct_ukraine_context"] | data["has_russia_context"]
    ) & data["has_crisis_financial_consequence"]
    data["geopolitical_crisis_financial"] = geopolitical
    data["crisis_specific_financial_consequence"] = consequence
    data["new_relevant_candidate"] = geopolitical | consequence
    data["new_relevance_path"] = assign_new_path(data)
    data["decision_changed"] = (
        data["old_relevant_candidate"] != data["new_relevant_candidate"]
    )
    return data


def validate_comparison(raw: pd.DataFrame, data: pd.DataFrame) -> None:
    """Validate provenance, source preservation, baseline, and refined logic."""

    if len(data) != EXPECTED_POSTS or data["id"].duplicated().any():
        raise ValueError("Comparison must contain all 3,033 unique posts.")
    if int(data["old_relevant_candidate"].sum()) != EXPECTED_OLD_RETAINED:
        raise ValueError("Phase 3B.2 baseline is not the audited 1,669 posts.")
    if not data["title"].equals(raw["title"]):
        raise ValueError("Comparison changed title text or raw row order.")
    if not data["selftext"].equals(raw["selftext"]):
        raise ValueError("Comparison changed selftext or raw row order.")
    if recompute_extraction_keywords("Russian stocks") != "Russian":
        raise ValueError("Bounded matching still confuses Russian with Russia.")
    if recompute_extraction_keywords("Russia stocks") != "Russia":
        raise ValueError("Bounded matching does not identify Russia.")
    if REFINED_FINANCIAL_PATTERN.search("Las Vegas casinos"):
        raise ValueError("Bounded gas matching incorrectly matched Vegas.")
    generic_russian_assets = [
        "Russian bonds",
        "Russian debt",
        "Russian assets",
        "Russian ADRs",
    ]
    if any(
        REFINED_CONSEQUENCE_PATTERN.search(text)
        for text in generic_russian_assets
    ):
        raise ValueError("A generic Russian asset phrase remains sufficient.")
    required_consequences = [
        "frozen assets",
        "assets frozen",
        "removed from SWIFT",
        "SWIFT sanctions",
        "divest from Russia",
        "oil embargo",
    ]
    if any(
        not REFINED_CONSEQUENCE_PATTERN.search(text)
        for text in required_consequences
    ):
        raise ValueError("A required crisis-specific consequence is missing.")
    if REFINED_CONSEQUENCE_PATTERN.search("SWIFT"):
        raise ValueError("Standalone SWIFT remains independently sufficient.")
    if not data["matched_keywords_recomputed"].equals(
        data["old_matched_keywords_recomputed"].astype(str)
    ):
        raise ValueError("Corrected Phase 3B.2 keyword provenance changed.")

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
    if not np.array_equal(expected, data["new_relevant_candidate"]):
        raise ValueError("The refined relevance formula is incorrect.")
    if not data.loc[data["title_only"], "full_text"].equals(
        data.loc[data["title_only"], "title"].astype(str)
    ):
        raise ValueError("Unavailable body text was used for classification.")


def deterministic_changed_sample(
    data: pd.DataFrame,
    newly_included: bool,
) -> pd.DataFrame:
    """Return all manageable changes or a balanced deterministic sample."""

    eligible = data.loc[
        data["old_relevant_candidate"].ne(newly_included)
        & data["new_relevant_candidate"].eq(newly_included)
    ].copy()
    eligible["change_type"] = (
        "newly_included" if newly_included else "newly_excluded"
    )
    if len(eligible) <= MAX_CHANGED_REVIEW_ROWS:
        chosen = eligible
    else:
        rng = np.random.default_rng(
            REDDIT_RULE_COMPARISON_RANDOM_SEED + (0 if newly_included else 1)
        )
        queues: dict[tuple[int, str], list[int]] = {}
        for stratum, group in eligible.groupby(
            ["year", "subreddit"], sort=True
        ):
            queues[stratum] = rng.permutation(group.index.to_numpy()).tolist()
        strata = list(queues)
        rng.shuffle(strata)
        selected: list[int] = []
        while len(selected) < MAX_CHANGED_REVIEW_ROWS:
            for stratum in strata:
                if queues[stratum] and len(selected) < MAX_CHANGED_REVIEW_ROWS:
                    selected.append(queues[stratum].pop())
        chosen = eligible.loc[selected].copy()
    chosen = chosen.sort_values(
        ["date_utc", "subreddit", "id"], kind="stable"
    ).reset_index(drop=True)
    expected_rows = min(len(eligible), MAX_CHANGED_REVIEW_ROWS)
    if len(chosen) != expected_rows or chosen["id"].duplicated().any():
        raise ValueError("Changed-decision review sample is invalid.")
    return chosen.loc[:, REVIEW_COLUMNS]


def write_outputs(
    data: pd.DataFrame,
    newly_included_review: pd.DataFrame,
    newly_excluded_review: pd.DataFrame,
) -> None:
    """Write diagnostics only; never create the final cleaned corpus."""

    REDDIT_RELEVANCE_RULE_COMPARISON_FILE.parent.mkdir(
        parents=True, exist_ok=True
    )
    data.loc[:, COMPARISON_COLUMNS].to_csv(
        REDDIT_RELEVANCE_RULE_COMPARISON_FILE,
        index=False,
        encoding="utf-8-sig",
    )
    newly_included_review.to_csv(
        REDDIT_RULE_NEWLY_INCLUDED_REVIEW_FILE,
        index=False,
        encoding="utf-8-sig",
    )
    newly_excluded_review.to_csv(
        REDDIT_RULE_NEWLY_EXCLUDED_REVIEW_FILE,
        index=False,
        encoding="utf-8-sig",
    )


def print_diagnostics(data: pd.DataFrame) -> None:
    """Print totals, composition, and decisions for the known cases."""

    retained = data.loc[data["new_relevant_candidate"]]
    newly_included = data.loc[
        ~data["old_relevant_candidate"] & data["new_relevant_candidate"]
    ]
    newly_excluded = data.loc[
        data["old_relevant_candidate"] & ~data["new_relevant_candidate"]
    ]
    print(f"Raw candidate posts: {len(data):,}")
    print(f"Old retained: {int(data['old_relevant_candidate'].sum()):,}")
    print(f"New retained: {len(retained):,}")
    print(f"New excluded: {len(data) - len(retained):,}")
    print(f"New retention percentage: {len(retained) / len(data):.2%}")
    print(f"Difference from old retained: {len(retained) - EXPECTED_OLD_RETAINED:+,}")
    print(f"Newly included: {len(newly_included):,}")
    print(f"Newly excluded: {len(newly_excluded):,}")
    print("Retained by year:", retained.groupby("year")["id"].nunique().to_dict())
    print(
        "Retained by subreddit:",
        retained.groupby("subreddit")["id"].nunique().to_dict(),
    )
    print(
        "Retained by new relevance path:",
        retained.groupby("new_relevance_path")["id"].nunique().to_dict(),
    )
    print(
        "Retained without direct Ukraine context:",
        int((~retained["has_direct_ukraine_context"]).sum()),
    )
    print(f"Title-only retained: {int(retained['title_only'].sum()):,}")
    print(
        "Retained title-only body states:",
        retained.loc[retained["title_only"]]
        .groupby("body_status")["id"]
        .nunique()
        .to_dict(),
    )
    print("Known borderline cases:")
    indexed = data.set_index("id")
    for post_id, description in KNOWN_CASES.items():
        if post_id not in indexed.index:
            print(f"  {post_id}: missing ({description})")
            continue
        row = indexed.loc[post_id]
        print(
            f"  {post_id}: old={bool(row['old_relevant_candidate'])}, "
            f"new={bool(row['new_relevant_candidate'])}, "
            f"path={row['new_relevance_path']} ({description})"
        )


def main() -> None:
    """Run the Phase 3B.3 comparison without finalizing the Reddit sample."""

    raw_hash_before = file_sha256(RAW_REDDIT_FILE)
    raw, old = load_inputs()
    data = build_comparison(raw, old)
    validate_comparison(raw, data)
    newly_included_review = deterministic_changed_sample(data, True)
    newly_excluded_review = deterministic_changed_sample(data, False)
    write_outputs(data, newly_included_review, newly_excluded_review)
    if file_sha256(RAW_REDDIT_FILE) != raw_hash_before:
        raise RuntimeError("The immutable Phase 3A raw corpus changed.")
    print_diagnostics(data)
    print(f"Saved comparison to {REDDIT_RELEVANCE_RULE_COMPARISON_FILE}")
    print(f"Saved newly included review to {REDDIT_RULE_NEWLY_INCLUDED_REVIEW_FILE}")
    print(f"Saved newly excluded review to {REDDIT_RULE_NEWLY_EXCLUDED_REVIEW_FILE}")


if __name__ == "__main__":
    main()
