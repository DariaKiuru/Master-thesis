"""Phase 4A: produce one FinBERT sentiment observation per Reddit post.

Why this phase exists
    The thesis needs a finance-domain measure of sentiment for each of the
    1,503 cleaned Ukraine-war-related posts before any daily aggregation.

Main input
    ``data/processed/reddit_posts_cleaned.csv`` with title/body-derived
    ``finbert_text`` for the frozen Phase 3B sample.

Main outputs
    ``data/processed/reddit_posts_finbert.csv`` with positive, neutral, and
    negative probabilities, ``sentiment_score``, a descriptive class label,
    chunk metadata, and Phase 4A diagnostics/review samples.

Methodological rules and boundaries
    The pretrained ``ProsusAI/finbert`` model is used without fine-tuning.
    Text is divided into approximately 30-word conceptual chunks, capped at
    120 per post. Any tokenizer-limit safeguard fragments are first recombined
    into one conceptual-chunk probability vector; conceptual chunks are then
    equally averaged so implementation safeguards do not give long text extra
    weight. Post sentiment is positive probability minus negative probability
    and lies in [-1, 1]; labels are probability argmax descriptions only. This
    expensive script does not construct daily or trading-day sentiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# Allow ``python src/07_score_finbert.py`` from the repository root.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from config import (  # noqa: E402
    CHUNK_MAX_WORDS,
    CLEANED_REDDIT_FILE,
    END_DATE,
    FINBERT_BATCH_SIZE,
    FINBERT_CHECKPOINT_DIR,
    FINBERT_CHECKPOINT_INTERVAL,
    FINBERT_CLASS_DISTRIBUTION_FILE,
    FINBERT_MODEL,
    FINBERT_REDDIT_FILE,
    FINBERT_REVIEW_SAMPLE_FILE,
    FINBERT_SENTIMENT_SUMMARY_FILE,
    MAX_CHUNKS_PER_POST,
    START_DATE,
    SUBREDDITS,
)


EXPECTED_INPUT_ROWS = 1_503
EXPECTED_RETAINED_CONCEPTUAL_CHUNKS = 34_879
EXPECTED_MODEL_INPUTS = 34_884
EXPECTED_FRAGMENTED_CONCEPTUAL_CHUNKS = 3
EXPECTED_INTERNAL_MODEL_FRAGMENTS = 8
EXPECTED_INPUT_SHA256 = (
    "9FA593C18509280231547E928F88003C5918BDEDD264DFAC3E0F8D0BDA46AC8D"
)
PROBABILITY_TOLERANCE = 1e-5
SCORE_TOLERANCE = 1e-10
REVIEW_EXAMPLES_PER_GROUP = 3
FULL_CHECKPOINT_FILE = FINBERT_CHECKPOINT_DIR / "full_probabilities.csv"
FULL_CHECKPOINT_METADATA_FILE = FINBERT_CHECKPOINT_DIR / "full_metadata.json"
CHECKPOINT_KEY_COLUMNS = [
    "post_position",
    "id",
    "chunk_number",
    "fragment_number",
]
CHECKPOINT_COLUMNS = ["logical_position", *CHECKPOINT_KEY_COLUMNS]
CHECKPOINT_PROBABILITY_COLUMNS = [
    "positive_probability",
    "neutral_probability",
    "negative_probability",
]

PROBABILITY_COLUMNS = [
    "post_positive_probability",
    "post_neutral_probability",
    "post_negative_probability",
]
PHASE4A_COLUMNS = [
    *PROBABILITY_COLUMNS,
    "sentiment_score",
    "sentiment_label",
    "n_chunks",
    "chunk_cap_applied",
    "tokenizer_safeguard_applied",
]
REVIEW_COLUMNS = [
    "review_group",
    "id",
    "date_utc",
    "subreddit",
    "title",
    "finbert_text",
    "title_only",
    "body_status",
    *PROBABILITY_COLUMNS,
    "sentiment_score",
    "sentiment_label",
    "n_chunks",
    "chunk_cap_applied",
    "tokenizer_safeguard_applied",
]
REVIEW_GROUPS = [
    "strongly_positive",
    "mildly_positive",
    "near_neutral",
    "mildly_negative",
    "strongly_negative",
    "high_neutral_probability",
    "title_only",
    "long_multi_chunk",
    "tokenizer_safeguard",
]
SUMMARY_COLUMNS = [
    "section",
    "grouping",
    "group",
    "metric",
    "value",
    "count",
    "percentage",
    "reddit_id",
    "conceptual_chunk_number",
    "whitespace_word_number",
    "original_word_character_length",
    "original_token_piece_count",
]
SENTIMENT_LABELS = ["positive", "neutral", "negative"]
SENTENCE_BOUNDARY_PATTERN = re.compile(r"(?<=[.!?])\s+")
WHITESPACE_PATTERN = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Validate the frozen post sample and construct conceptual chunks
# ---------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    """Parse the small set of Phase 4A execution controls."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun inference even when complete valid Phase 4A files exist.",
    )
    parser.add_argument(
        "--validate-input-only",
        action="store_true",
        help="Validate the frozen Phase 3B input without loading FinBERT.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the validated full-run probability checkpoint.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=FINBERT_BATCH_SIZE,
        help="Computational batch size; default is the benchmarked value 8.",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=FINBERT_CHECKPOINT_INTERVAL,
        help="Atomically checkpoint after this many completed model inputs.",
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    """Return an uppercase SHA-256 checksum for one file."""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def validate_configuration() -> None:
    """Confirm the locked Phase 4A methodological settings."""

    if FINBERT_MODEL != "ProsusAI/finbert":
        raise ValueError("Phase 4A must use ProsusAI/finbert.")
    if CHUNK_MAX_WORDS != 30 or MAX_CHUNKS_PER_POST != 120:
        raise ValueError("Phase 4A requires 30-word chunks and a 120-chunk cap.")
    if FINBERT_BATCH_SIZE < 1:
        raise ValueError("FINBERT_BATCH_SIZE must be positive.")
    if FINBERT_CHECKPOINT_INTERVAL < 1:
        raise ValueError("FINBERT_CHECKPOINT_INTERVAL must be positive.")
    if START_DATE != "2021-01-01" or END_DATE != "2023-12-31":
        raise ValueError("Phase 4A must cover 2021-01-01 through 2023-12-31.")
    if SUBREDDITS != ["investing", "stocks", "StockMarket"]:
        raise ValueError("The configured subreddit universe is incorrect.")


def load_and_validate_input() -> tuple[pd.DataFrame, str]:
    """Load the frozen cleaned corpus and validate its identity and contents."""

    if not CLEANED_REDDIT_FILE.exists():
        raise FileNotFoundError(f"Cleaned Reddit input not found: {CLEANED_REDDIT_FILE}")
    input_sha256 = file_sha256(CLEANED_REDDIT_FILE)
    if input_sha256 != EXPECTED_INPUT_SHA256:
        raise RuntimeError(
            "Frozen Phase 3B input checksum differs from the validated value: "
            f"expected {EXPECTED_INPUT_SHA256}, found {input_sha256}."
        )

    data = pd.read_csv(
        CLEANED_REDDIT_FILE,
        dtype={"id": "string", "subreddit": "string"},
        keep_default_na=False,
    )
    required_columns = {
        "id",
        "created_utc",
        "created_datetime_utc",
        "date_utc",
        "subreddit",
        "title",
        "selftext",
        "full_text",
        "finbert_text",
        "body_status",
        "title_only",
        "relevance_path",
        "language_status",
    }
    if missing := required_columns - set(data.columns):
        raise ValueError(f"Cleaned Reddit input lacks columns: {sorted(missing)}")
    if len(data) != EXPECTED_INPUT_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_INPUT_ROWS:,} cleaned posts, found {len(data):,}."
        )
    if data["id"].astype(str).str.strip().eq("").any():
        raise ValueError("At least one cleaned Reddit post has a blank ID.")
    if data["id"].duplicated().any():
        raise ValueError("Cleaned Reddit IDs must be unique.")
    if data["finbert_text"].astype(str).str.strip().eq("").any():
        raise ValueError("At least one cleaned post has blank finbert_text.")
    dates = pd.to_datetime(data["date_utc"], errors="coerce")
    if dates.isna().any() or not dates.between(
        START_DATE, END_DATE, inclusive="both"
    ).all():
        raise ValueError("Cleaned Reddit input contains an invalid sample date.")
    if set(data["subreddit"].unique()) != set(SUBREDDITS):
        raise ValueError("Cleaned Reddit input contains an unexpected subreddit.")
    return data, input_sha256


def parse_boolean_series(series: pd.Series, column_name: str) -> pd.Series:
    """Parse a Boolean-like series without treating non-empty strings as true."""

    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.casefold()
    unexpected = set(normalized.unique()) - {"true", "false"}
    if unexpected:
        raise ValueError(
            f"Unexpected Boolean values in {column_name}: {sorted(unexpected)}"
        )
    return normalized.eq("true")


def split_into_chunks(text: str) -> list[str]:
    """Split one post into ordered, sentence-aware conceptual chunks.

    The target is approximately 30 whitespace-delimited words. Every word is
    retained in order, and only an individual sentence longer than the target
    is divided into consecutive pieces.
    """

    normalized = WHITESPACE_PATTERN.sub(" ", str(text)).strip()
    if not normalized:
        return []
    sentences = SENTENCE_BOUNDARY_PATTERN.split(normalized)
    chunks: list[str] = []
    current_sentences: list[str] = []
    current_word_count = 0

    for sentence in sentences:
        sentence_words = sentence.split()
        word_count = len(sentence_words)

        # Preserve ordinary sentences. Only a sentence that cannot fit within
        # the locked word target is split into consecutive, non-overlapping
        # segments, with every word retained in its original order.
        if word_count > CHUNK_MAX_WORDS:
            if current_sentences:
                chunks.append(" ".join(current_sentences))
                current_sentences = []
                current_word_count = 0
            for start in range(0, word_count, CHUNK_MAX_WORDS):
                chunks.append(
                    " ".join(sentence_words[start : start + CHUNK_MAX_WORDS])
                )
            continue

        if current_sentences and current_word_count + word_count > CHUNK_MAX_WORDS:
            chunks.append(" ".join(current_sentences))
            current_sentences = [sentence]
            current_word_count = word_count
        else:
            current_sentences.append(sentence)
            current_word_count += word_count
    if current_sentences:
        chunks.append(" ".join(current_sentences))
    if any(len(chunk.split()) > CHUNK_MAX_WORDS for chunk in chunks):
        raise ValueError("Deterministic chunking produced a chunk over 30 words.")
    if " ".join(" ".join(chunks).split()) != normalized:
        raise ValueError("Deterministic chunking changed text content or word order.")
    return chunks


def build_chunk_table(posts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create the complete ordered uncapped chunks and per-post cap metadata."""

    chunk_rows: list[dict[str, Any]] = []
    post_rows: list[dict[str, Any]] = []
    for post_position, row in enumerate(posts.itertuples(index=False)):
        post_id = str(row.id)
        uncapped_chunks = split_into_chunks(str(row.finbert_text))
        if not uncapped_chunks:
            raise ValueError(f"Post {post_id} produced no FinBERT chunks.")
        cap_applied = len(uncapped_chunks) > MAX_CHUNKS_PER_POST
        post_rows.append(
            {
                "post_position": post_position,
                "id": post_id,
                "uncapped_n_chunks": len(uncapped_chunks),
                "n_chunks": min(len(uncapped_chunks), MAX_CHUNKS_PER_POST),
                "chunk_cap_applied": cap_applied,
            }
        )
        for chunk_number, chunk_text in enumerate(uncapped_chunks, start=1):
            chunk_rows.append(
                {
                    "post_position": post_position,
                    "id": post_id,
                    "chunk_number": chunk_number,
                    "chunk_text": chunk_text,
                }
            )

    chunks = pd.DataFrame(chunk_rows)
    post_metadata = pd.DataFrame(post_rows)
    if not post_metadata["n_chunks"].between(1, MAX_CHUNKS_PER_POST).all():
        raise ValueError("A post has an invalid number of chunks to score.")
    expected_cap = post_metadata["uncapped_n_chunks"].gt(MAX_CHUNKS_PER_POST)
    if not np.array_equal(expected_cap, post_metadata["chunk_cap_applied"]):
        raise ValueError("chunk_cap_applied does not reflect actual truncation.")
    return chunks, post_metadata


def apply_chunk_cap(
    uncapped_chunks: pd.DataFrame,
    post_metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Retain the first 120 tokenizer-safe chunks per post in original order."""

    scored_chunks = uncapped_chunks.loc[
        uncapped_chunks["chunk_number"].le(MAX_CHUNKS_PER_POST)
    ].copy()
    expected_scored_chunks = int(post_metadata["n_chunks"].sum())
    if len(scored_chunks) != expected_scored_chunks:
        raise ValueError("The scored chunk count does not match the post metadata.")
    if not scored_chunks.groupby("post_position", sort=False)["chunk_number"].apply(
        lambda values: values.tolist() == list(range(1, len(values) + 1))
    ).all():
        raise ValueError("The chunk cap changed chunk order or created a gap.")
    return scored_chunks.reset_index(drop=True)


def normalize_label_mapping(model: Any) -> dict[int, str]:
    """Resolve the model's actual output indices to the three FinBERT labels."""

    raw_mapping = getattr(model.config, "id2label", None)
    if not isinstance(raw_mapping, dict) or not raw_mapping:
        raise ValueError("FinBERT model config does not expose a usable id2label map.")
    mapping: dict[int, str] = {}
    for raw_index, raw_label in raw_mapping.items():
        try:
            index = int(raw_index)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid FinBERT class index: {raw_index!r}") from error
        label = str(raw_label).strip().casefold()
        if label not in SENTIMENT_LABELS:
            raise ValueError(f"Unexpected FinBERT class label: {raw_label!r}")
        if index in mapping or label in mapping.values():
            raise ValueError("FinBERT id2label mapping is ambiguous.")
        mapping[index] = label
    if set(mapping.values()) != set(SENTIMENT_LABELS) or len(mapping) != 3:
        raise ValueError(
            "FinBERT label mapping must contain exactly positive, neutral, negative."
        )

    raw_reverse = getattr(model.config, "label2id", None)
    if isinstance(raw_reverse, dict) and raw_reverse:
        normalized_reverse = {
            str(label).strip().casefold(): int(index)
            for label, index in raw_reverse.items()
        }
        expected_reverse = {label: index for index, label in mapping.items()}
        if normalized_reverse != expected_reverse:
            raise ValueError("FinBERT id2label and label2id mappings disagree.")
    return mapping


def load_finbert() -> tuple[Any, Any, dict[int, str], dict[str, str], Any]:
    """Load the locked pretrained model and return runtime provenance.

    ``model.eval()`` selects inference behavior only; no training or
    thesis-specific fine-tuning occurs.
    """

    import torch
    import transformers
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL)
    label_mapping = normalize_label_mapping(model)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    reported_identifier = str(
        getattr(model.config, "_name_or_path", "")
        or getattr(model.config, "name_or_path", "")
        or FINBERT_MODEL
    )
    revision = str(
        getattr(model.config, "_commit_hash", "")
        or getattr(tokenizer, "init_kwargs", {}).get("_commit_hash", "")
        or ""
    )
    runtime = {
        "model_identifier": FINBERT_MODEL,
        "model_reported_identifier": reported_identifier,
        "model_revision": revision,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "inference_device": str(device),
    }
    return tokenizer, model, label_mapping, runtime, torch


def supported_input_length(tokenizer: Any, model: Any) -> int:
    """Resolve the finite model input limit used only as a validation guard."""

    candidates: list[int] = []
    tokenizer_limit = getattr(tokenizer, "model_max_length", None)
    if isinstance(tokenizer_limit, int) and 0 < tokenizer_limit < 1_000_000:
        candidates.append(tokenizer_limit)
    model_limit = getattr(model.config, "max_position_embeddings", None)
    if isinstance(model_limit, int) and model_limit > 0:
        candidates.append(model_limit)
    if not candidates:
        raise ValueError("Could not determine FinBERT's supported input length.")
    return min(candidates)


def split_into_model_fragments(
    inference_words: list[str],
    tokenizer: Any,
    maximum_length: int,
) -> list[list[str]]:
    """Greedily form the largest consecutive word-boundary model inputs."""

    fragments: list[list[str]] = []
    current_words: list[str] = []
    for word in inference_words:
        candidate_words = [*current_words, word]
        candidate_length = len(
            tokenizer(
                " ".join(candidate_words),
                add_special_tokens=True,
                truncation=False,
            )["input_ids"]
        )
        if candidate_length <= maximum_length:
            current_words = candidate_words
            continue
        if not current_words:
            raise ValueError(
                "A safeguarded whitespace token still cannot fit in one model input."
            )
        fragments.append(current_words)
        current_words = [word]
        single_word_length = len(
            tokenizer(
                word,
                add_special_tokens=True,
                truncation=False,
            )["input_ids"]
        )
        if single_word_length > maximum_length:
            raise ValueError(
                "A safeguarded whitespace token still exceeds the model limit."
            )
    if current_words:
        fragments.append(current_words)
    return fragments


def prepare_model_inputs(
    conceptual_chunks: pd.DataFrame,
    tokenizer: Any,
    model: Any,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Create tokenizer-safe model inputs beneath retained conceptual chunks.

    Safeguard fragments exist only to respect FinBERT's token limit. Their
    relationship to the parent conceptual chunk is retained so fragmentation
    cannot create extra weight in the post-level average.
    """

    maximum_length = supported_input_length(tokenizer, model)
    special_token_count = int(tokenizer.num_special_tokens_to_add(pair=False))
    usable_payload = maximum_length - special_token_count
    if special_token_count < 1 or usable_payload < 1:
        raise ValueError("The tokenizer/model payload capacity could not be resolved.")
    if not tokenizer.unk_token:
        raise ValueError("The FinBERT tokenizer does not expose an unknown token.")
    unk_piece_count = len(
        tokenizer(
            tokenizer.unk_token,
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]
    )
    if not 1 <= unk_piece_count <= usable_payload:
        raise ValueError("The tokenizer unknown token cannot be used safely.")

    initial_encodings = tokenizer(
        conceptual_chunks["chunk_text"].tolist(),
        add_special_tokens=True,
        truncation=False,
        padding=False,
    )["input_ids"]
    initial_lengths = [len(input_ids) for input_ids in initial_encodings]

    model_input_rows: list[dict[str, Any]] = []
    replacement_rows: list[dict[str, Any]] = []
    oversized_chunk_keys: set[tuple[int, int]] = set()
    unk_chunk_keys: set[tuple[int, int]] = set()
    fragmented_chunk_keys: set[tuple[int, int]] = set()
    internal_fragment_count = 0

    for row, initial_length in zip(
        conceptual_chunks.itertuples(index=False), initial_lengths
    ):
        conceptual_words = str(row.chunk_text).split()
        conceptual_word_count = len(conceptual_words)
        if conceptual_word_count < 1:
            raise ValueError(f"Post {row.id}, chunk {row.chunk_number} is blank.")
        key = (int(row.post_position), int(row.chunk_number))
        inference_words = conceptual_words.copy()
        safeguard_applied = False

        if initial_length > maximum_length:
            oversized_chunk_keys.add(key)
            individual_encodings = tokenizer(
                conceptual_words,
                add_special_tokens=False,
                truncation=False,
                padding=False,
            )["input_ids"]
            for word_position, (word, word_ids) in enumerate(
                zip(conceptual_words, individual_encodings), start=1
            ):
                if len(word_ids) > usable_payload:
                    inference_words[word_position - 1] = tokenizer.unk_token
                    safeguard_applied = True
                    unk_chunk_keys.add(key)
                    replacement_rows.append(
                        {
                            "reddit_id": str(row.id),
                            "post_position": int(row.post_position),
                            "conceptual_chunk_number": int(row.chunk_number),
                            "whitespace_word_number": word_position,
                            "original_word_character_length": len(word),
                            "original_token_piece_count": len(word_ids),
                        }
                    )

        inference_text = " ".join(inference_words)
        safeguarded_length = len(
            tokenizer(
                inference_text,
                add_special_tokens=True,
                truncation=False,
            )["input_ids"]
        )
        if safeguarded_length <= maximum_length:
            fragments = [inference_words]
        else:
            safeguard_applied = True
            fragmented_chunk_keys.add(key)
            fragments = split_into_model_fragments(
                inference_words, tokenizer, maximum_length
            )
            internal_fragment_count += len(fragments)

        for fragment_number, fragment_words in enumerate(fragments, start=1):
            model_input_rows.append(
                {
                    "post_position": int(row.post_position),
                    "id": str(row.id),
                    "chunk_number": int(row.chunk_number),
                    "fragment_number": fragment_number,
                    "model_input_text": " ".join(fragment_words),
                    "fragment_word_count": len(fragment_words),
                    "conceptual_chunk_word_count": conceptual_word_count,
                    "tokenizer_safeguard_applied": safeguard_applied,
                }
            )

    model_inputs = pd.DataFrame(model_input_rows)
    replacements = pd.DataFrame(
        replacement_rows,
        columns=[
            "reddit_id",
            "post_position",
            "conceptual_chunk_number",
            "whitespace_word_number",
            "original_word_character_length",
            "original_token_piece_count",
        ],
    )
    final_encodings = tokenizer(
        model_inputs["model_input_text"].tolist(),
        add_special_tokens=True,
        truncation=False,
        padding=False,
    )["input_ids"]
    final_lengths = [len(input_ids) for input_ids in final_encodings]
    excessive_positions = [
        position
        for position, token_length in enumerate(final_lengths)
        if token_length > maximum_length
    ]
    statistics = {
        "retained_conceptual_chunks_checked": len(conceptual_chunks),
        "maximum_retained_token_length_before_safeguard": max(initial_lengths),
        "oversized_retained_conceptual_chunks_before_safeguard": len(
            oversized_chunk_keys
        ),
        "conceptual_chunks_requiring_unk": len(unk_chunk_keys),
        "impossible_whitespace_tokens_replaced_with_unk": len(replacements),
        "conceptual_chunks_requiring_internal_fragmentation": len(
            fragmented_chunk_keys
        ),
        "total_internal_model_fragments": internal_fragment_count,
        "total_final_model_inputs": len(model_inputs),
        "additional_model_inputs_from_internal_fragmentation": (
            len(model_inputs) - len(conceptual_chunks)
        ),
        "model_input_special_tokens": special_token_count,
        "model_usable_payload_tokens": usable_payload,
        "maximum_final_model_input_token_length": max(final_lengths),
        "final_model_inputs_exceeding_limit": len(excessive_positions),
        "model_token_limit": maximum_length,
    }
    expected_additional_inputs = internal_fragment_count - len(
        fragmented_chunk_keys
    )
    if (
        statistics["additional_model_inputs_from_internal_fragmentation"]
        != expected_additional_inputs
    ):
        raise ValueError("Internal-fragment model inputs do not reconcile.")
    print(
        "Tokenizer safeguard | conceptual chunks="
        f"{statistics['retained_conceptual_chunks_checked']:,} | requiring [UNK]="
        f"{statistics['conceptual_chunks_requiring_unk']:,} | requiring internal "
        f"fragmentation={statistics['conceptual_chunks_requiring_internal_fragmentation']:,} "
        f"| maximum final tokens={statistics['maximum_final_model_input_token_length']:,} "
        f"| still exceeding={statistics['final_model_inputs_exceeding_limit']:,}",
        flush=True,
    )
    if excessive_positions:
        cases = []
        for position in excessive_positions[:10]:
            row = model_inputs.iloc[position]
            cases.append(
                f"ID={row['id']}, chunk={int(row['chunk_number'])}, "
                f"fragment={int(row['fragment_number'])}, "
                f"words={int(row['fragment_word_count'])}, "
                f"tokens={final_lengths[position]}"
            )
        raise ValueError(
            "Tokenizer safeguard left model inputs above the supported limit: "
            + "; ".join(cases)
        )
    expected_words = model_inputs.groupby(
        ["post_position", "chunk_number"], sort=False
    )["fragment_word_count"].sum()
    conceptual_words = conceptual_chunks.set_index(
        ["post_position", "chunk_number"]
    )["chunk_text"].map(lambda text: len(str(text).split()))
    if not expected_words.equals(conceptual_words):
        raise ValueError("Tokenizer safeguard changed conceptual word counts.")
    return model_inputs, replacements, statistics


def model_input_fingerprint(
    model_inputs: pd.DataFrame,
    input_sha256: str,
    model_identifier: str,
    model_revision: str,
) -> str:
    """Identify the exact ordered model inputs behind a recoverable checkpoint."""

    digest = hashlib.sha256()
    digest.update(
        f"{input_sha256}|{model_identifier}|{model_revision}|"
        f"{len(model_inputs)}".encode("utf-8")
    )
    for row in model_inputs.itertuples(index=False):
        digest.update(
            f"\n{row.post_position}|{row.id}|{row.chunk_number}|"
            f"{row.fragment_number}|{row.model_input_text}".encode("utf-8")
        )
    return digest.hexdigest().upper()


def build_length_aware_processing_order(
    model_inputs: pd.DataFrame,
    tokenizer: Any,
) -> tuple[list[int], list[int]]:
    """Sort deterministically by exact token length and stable logical position."""

    encoded = tokenizer(
        model_inputs["model_input_text"].tolist(),
        add_special_tokens=True,
        truncation=False,
        padding=False,
    )["input_ids"]
    token_lengths = [len(input_ids) for input_ids in encoded]
    processing_order = sorted(
        range(len(model_inputs)),
        key=lambda position: (token_lengths[position], position),
    )
    if sorted(processing_order) != list(range(len(model_inputs))):
        raise ValueError("Length-aware processing order is not a full permutation.")
    return processing_order, token_lengths


def write_checkpoint_atomic(
    model_inputs: pd.DataFrame,
    processing_order: list[int],
    probability_rows: list[dict[str, float]],
) -> None:
    """Atomically persist the completed processing prefix and probabilities."""

    completed_positions = processing_order[: len(probability_rows)]
    checkpoint = model_inputs.iloc[completed_positions].loc[
        :, CHECKPOINT_KEY_COLUMNS
    ].reset_index(drop=True)
    checkpoint.insert(0, "logical_position", completed_positions)
    checkpoint = pd.concat(
        [
            checkpoint,
            pd.DataFrame(
                probability_rows,
                columns=CHECKPOINT_PROBABILITY_COLUMNS,
            ),
        ],
        axis="columns",
    )
    FULL_CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = FULL_CHECKPOINT_FILE.with_suffix(".tmp")
    checkpoint.to_csv(temporary_file, index=False)
    temporary_file.replace(FULL_CHECKPOINT_FILE)


def initialize_or_validate_checkpoint(
    model_inputs: pd.DataFrame,
    processing_order: list[int],
    metadata: dict[str, Any],
    resume: bool,
) -> list[dict[str, float]]:
    """Load only an exact validated processing prefix or initialize a new run."""

    if not resume:
        if FULL_CHECKPOINT_FILE.exists() or FULL_CHECKPOINT_METADATA_FILE.exists():
            raise FileExistsError(
                "A full Phase 4A checkpoint already exists. Use --resume rather "
                "than restarting, or move the checkpoint after review."
            )
        FULL_CHECKPOINT_METADATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        temporary_metadata = FULL_CHECKPOINT_METADATA_FILE.with_suffix(".tmp")
        temporary_metadata.write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        temporary_metadata.replace(FULL_CHECKPOINT_METADATA_FILE)
        return []

    if not FULL_CHECKPOINT_METADATA_FILE.exists():
        raise FileNotFoundError("Full checkpoint metadata is required to resume.")
    recorded_metadata = json.loads(
        FULL_CHECKPOINT_METADATA_FILE.read_text(encoding="utf-8")
    )
    if recorded_metadata != metadata:
        raise ValueError("Full checkpoint metadata does not match this production run.")
    if not FULL_CHECKPOINT_FILE.exists():
        print("Checkpoint metadata is valid; no probability block was completed yet.")
        return []

    checkpoint = pd.read_csv(FULL_CHECKPOINT_FILE, dtype={"id": "string"})
    expected_columns = [*CHECKPOINT_COLUMNS, *CHECKPOINT_PROBABILITY_COLUMNS]
    if list(checkpoint.columns) != expected_columns:
        raise ValueError("Full checkpoint schema is invalid.")
    if len(checkpoint) > len(model_inputs):
        raise ValueError("Full checkpoint contains too many model inputs.")

    expected_positions = processing_order[: len(checkpoint)]
    recorded_positions = pd.to_numeric(
        checkpoint["logical_position"], errors="coerce"
    )
    if recorded_positions.isna().any() or not np.array_equal(
        recorded_positions.to_numpy(dtype=int), expected_positions
    ):
        raise ValueError("Full checkpoint processing order is not the expected prefix.")
    current_keys = model_inputs.iloc[expected_positions].loc[
        :, CHECKPOINT_KEY_COLUMNS
    ].reset_index(drop=True)
    checkpoint_keys = checkpoint.loc[:, CHECKPOINT_KEY_COLUMNS].copy()
    for column in ["post_position", "chunk_number", "fragment_number"]:
        current_keys[column] = pd.to_numeric(current_keys[column]).astype(int)
        checkpoint_keys[column] = pd.to_numeric(checkpoint_keys[column]).astype(int)
    current_keys["id"] = current_keys["id"].astype("string")
    checkpoint_keys["id"] = checkpoint_keys["id"].astype("string")
    if not current_keys.equals(checkpoint_keys.reset_index(drop=True)):
        raise ValueError("Full checkpoint keys do not match the expected input prefix.")
    validate_chunk_probabilities(checkpoint)
    return checkpoint.loc[:, CHECKPOINT_PROBABILITY_COLUMNS].to_dict("records")


def format_duration(seconds: float) -> str:
    """Format elapsed and remaining durations compactly for progress output."""

    if not np.isfinite(seconds) or seconds < 0:
        return "unknown"
    total_seconds = int(round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def score_model_inputs(
    model_inputs: pd.DataFrame,
    tokenizer: Any,
    model: Any,
    label_mapping: dict[int, str],
    torch: Any,
    processing_order: list[int],
    *,
    batch_size: int,
    checkpoint_interval: int,
    probability_rows: list[dict[str, float]],
) -> pd.DataFrame:
    """Run length-aware batched inference with atomic recoverable checkpoints."""

    if batch_size < 1 or checkpoint_interval < 1:
        raise ValueError("Batch size and checkpoint interval must be positive.")
    completed_at_start = len(probability_rows)
    device = next(model.parameters()).device
    started = time.perf_counter()
    next_checkpoint = (
        (completed_at_start // checkpoint_interval) + 1
    ) * checkpoint_interval

    for start in range(completed_at_start, len(model_inputs), batch_size):
        batch_positions = processing_order[start : start + batch_size]
        batch = model_inputs.iloc[batch_positions]
        try:
            encoded = tokenizer(
                batch["model_input_text"].tolist(),
                padding=True,
                truncation=False,
                return_tensors="pt",
            )
            encoded = {name: values.to(device) for name, values in encoded.items()}
            with torch.inference_mode():
                logits = model(**encoded).logits
                probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
        except Exception as error:
            locations = ", ".join(
                f"{row.id}:chunk{row.chunk_number}:fragment{row.fragment_number}"
                for row in batch.itertuples(index=False)
            )
            raise RuntimeError(
                f"FinBERT inference failed for model inputs {locations}."
            ) from error

        if probabilities.shape != (len(batch), len(label_mapping)):
            raise ValueError(f"Unexpected probability shape: {probabilities.shape}.")
        if not np.isfinite(probabilities).all() or not np.allclose(
            probabilities.sum(axis=1), 1.0, atol=PROBABILITY_TOLERANCE, rtol=0
        ):
            raise ValueError("A batch produced invalid softmax probabilities.")
        for values in probabilities:
            probability_rows.append(
                {
                    f"{label_mapping[index]}_probability": float(values[index])
                    for index in label_mapping
                }
            )

        completed = len(probability_rows)
        if completed >= next_checkpoint or completed == len(model_inputs):
            write_checkpoint_atomic(model_inputs, processing_order, probability_rows)
            elapsed = time.perf_counter() - started
            newly_scored = completed - completed_at_start
            throughput = newly_scored / elapsed if elapsed else float("nan")
            remaining = len(model_inputs) - completed
            eta_seconds = remaining / throughput if throughput > 0 else float("nan")
            percentage = completed / len(model_inputs) * 100
            print(
                f"Checkpoint | {completed:,}/{len(model_inputs):,} "
                f"({percentage:.2f}%) | elapsed={format_duration(elapsed)} | "
                f"throughput={throughput:.2f} inputs/s | "
                f"ETA={format_duration(eta_seconds)} | {FULL_CHECKPOINT_FILE}",
                flush=True,
            )
            while next_checkpoint <= completed:
                next_checkpoint += checkpoint_interval

    if len(probability_rows) != len(model_inputs):
        raise ValueError("Production inference did not score every model input.")
    ordered_probability_rows: list[dict[str, float] | None] = [
        None
    ] * len(model_inputs)
    for logical_position, probabilities in zip(processing_order, probability_rows):
        ordered_probability_rows[logical_position] = probabilities
    if any(row is None for row in ordered_probability_rows):
        raise ValueError("Logical model-input order was not fully restored.")
    result = model_inputs.copy()
    ordered_probabilities = pd.DataFrame(
        ordered_probability_rows,
        columns=CHECKPOINT_PROBABILITY_COLUMNS,
    )
    for column in CHECKPOINT_PROBABILITY_COLUMNS:
        result[column] = ordered_probabilities[column].to_numpy()
    validate_chunk_probabilities(result)
    return result


def reconstruct_conceptual_probabilities(
    scored_model_inputs: pd.DataFrame,
    conceptual_chunks: pd.DataFrame,
) -> pd.DataFrame:
    """Reconstruct one probability vector per conceptual chunk.

    When a chunk required several model inputs, fragment probabilities are
    combined using their shares of the chunk's words. The returned table again
    has exactly one positive/neutral/negative vector per conceptual chunk.
    """

    data = scored_model_inputs.copy()
    data["fragment_weight"] = (
        data["fragment_word_count"] / data["conceptual_chunk_word_count"]
    )
    weighted_columns: list[str] = []
    for label in SENTIMENT_LABELS:
        weighted_column = f"weighted_{label}_probability"
        data[weighted_column] = data[f"{label}_probability"] * data["fragment_weight"]
        weighted_columns.append(weighted_column)
    group_columns = ["post_position", "id", "chunk_number"]
    probabilities = data.groupby(group_columns, sort=False)[weighted_columns].sum()
    probabilities.columns = [f"{label}_probability" for label in SENTIMENT_LABELS]
    safeguard = data.groupby(group_columns, sort=False)[
        "tokenizer_safeguard_applied"
    ].max()
    reconstructed = probabilities.join(safeguard).reset_index()
    expected_keys = conceptual_chunks.loc[:, group_columns].reset_index(drop=True)
    if not reconstructed.loc[:, group_columns].reset_index(drop=True).equals(
        expected_keys
    ):
        raise ValueError("Model fragments did not reconstruct every conceptual chunk.")
    validate_chunk_probabilities(reconstructed)
    return reconstructed


def validate_chunk_probabilities(chunks: pd.DataFrame) -> None:
    """Validate all chunk-level softmax probabilities.

    Values in [0, 1] that sum to one verify that the FinBERT outputs and any
    safeguard reconstruction still form valid probability distributions.
    """

    columns = [f"{label}_probability" for label in SENTIMENT_LABELS]
    values = chunks[columns].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Chunk-level probabilities contain a non-finite value.")
    if ((values < 0) | (values > 1)).any():
        raise ValueError("Chunk-level probabilities fall outside [0, 1].")
    if not np.allclose(values.sum(axis=1), 1.0, atol=PROBABILITY_TOLERANCE, rtol=0):
        raise ValueError("Chunk-level probabilities do not sum to one.")


def aggregate_posts(
    posts: pd.DataFrame,
    scored_chunks: pd.DataFrame,
    post_metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Average conceptual chunks to one equally weighted post observation.

    Positive, neutral, and negative probabilities are averaged separately.
    ``sentiment_score`` is positive minus negative, while the argmax label is
    retained only for descriptive class counts and uses no arbitrary threshold.
    """

    chunk_probability_columns = [
        "positive_probability",
        "neutral_probability",
        "negative_probability",
    ]
    averaged = (
        scored_chunks.groupby("post_position", sort=False)[chunk_probability_columns]
        .mean()
        .reindex(range(len(posts)))
    )
    if averaged.isna().any().any():
        raise ValueError("At least one post is missing aggregated probabilities.")
    observed_chunk_counts = scored_chunks.groupby("post_position", sort=False).size()
    expected_chunk_counts = post_metadata.set_index("post_position")["n_chunks"]
    if not observed_chunk_counts.equals(expected_chunk_counts):
        raise ValueError("Conceptual chunk counts changed before post aggregation.")

    output = posts.copy()
    output["post_positive_probability"] = averaged[
        "positive_probability"
    ].to_numpy()
    output["post_neutral_probability"] = averaged[
        "neutral_probability"
    ].to_numpy()
    output["post_negative_probability"] = averaged[
        "negative_probability"
    ].to_numpy()
    output["sentiment_score"] = (
        output["post_positive_probability"]
        - output["post_negative_probability"]
    )
    output["sentiment_label"] = output[PROBABILITY_COLUMNS].idxmax(axis=1).str.replace(
        r"^post_|_probability$", "", regex=True
    )
    ordered_metadata = post_metadata.sort_values("post_position")
    output["n_chunks"] = ordered_metadata["n_chunks"].to_numpy(dtype=int)
    output["chunk_cap_applied"] = ordered_metadata[
        "chunk_cap_applied"
    ].to_numpy(dtype=bool)
    safeguard_by_post = scored_chunks.groupby("post_position", sort=False)[
        "tokenizer_safeguard_applied"
    ].max().reindex(range(len(posts)), fill_value=False)
    output["tokenizer_safeguard_applied"] = safeguard_by_post.to_numpy(dtype=bool)
    return output


def validate_post_output(input_data: pd.DataFrame, output: pd.DataFrame) -> None:
    """Apply all locked Phase 4A post-level validation checks."""

    if len(output) != EXPECTED_INPUT_ROWS:
        raise ValueError(f"Phase 4A output has {len(output):,} rows.")
    if output["id"].nunique() != EXPECTED_INPUT_ROWS or output["id"].duplicated().any():
        raise ValueError("Phase 4A output IDs are not unique and complete.")
    if not output["id"].reset_index(drop=True).equals(
        input_data["id"].reset_index(drop=True)
    ):
        raise ValueError("Phase 4A output changed the input ID set or ordering.")
    original_columns = list(input_data.columns)
    if list(output.columns) != [*original_columns, *PHASE4A_COLUMNS]:
        raise ValueError("Phase 4A output schema is incorrect.")
    if not output.loc[:, original_columns].reset_index(drop=True).equals(
        input_data.reset_index(drop=True)
    ):
        raise ValueError("An original Phase 3B field changed in memory.")

    probabilities = output[PROBABILITY_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all():
        raise ValueError("Post probabilities contain a non-finite value.")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Post probabilities fall outside [0, 1].")
    if not np.allclose(
        probabilities.sum(axis=1), 1.0, atol=PROBABILITY_TOLERANCE, rtol=0
    ):
        raise ValueError("Post probabilities do not sum to one.")

    scores = output["sentiment_score"].to_numpy(dtype=float)
    if not np.isfinite(scores).all() or ((scores < -1) | (scores > 1)).any():
        raise ValueError("sentiment_score is non-finite or outside [-1, 1].")
    expected_scores = probabilities[:, 0] - probabilities[:, 2]
    if not np.allclose(scores, expected_scores, atol=SCORE_TOLERANCE, rtol=0):
        raise ValueError("sentiment_score does not equal positive minus negative.")

    labels = output["sentiment_label"].astype(str)
    if set(labels.unique()) - set(SENTIMENT_LABELS):
        raise ValueError("Phase 4A output contains an invalid sentiment label.")
    expected_labels = np.asarray(SENTIMENT_LABELS)[probabilities.argmax(axis=1)]
    if not np.array_equal(labels.to_numpy(), expected_labels):
        raise ValueError("sentiment_label does not equal probability argmax.")

    n_chunks = pd.to_numeric(output["n_chunks"], errors="coerce")
    if n_chunks.isna().any() or not n_chunks.between(1, MAX_CHUNKS_PER_POST).all():
        raise ValueError("n_chunks must be an integer from 1 through 120.")
    if not np.equal(n_chunks.to_numpy(), np.floor(n_chunks.to_numpy())).all():
        raise ValueError("n_chunks contains a non-integer value.")
    parse_boolean_series(output["chunk_cap_applied"], "chunk_cap_applied")
    parse_boolean_series(
        output["tokenizer_safeguard_applied"],
        "tokenizer_safeguard_applied",
    )


def add_summary_value(
    rows: list[dict[str, Any]],
    section: str,
    grouping: str,
    group: str,
    metric: str,
    value: Any,
    *,
    count: Any = "",
    percentage: Any = "",
) -> None:
    """Append one row to the readable long-form sentiment summary."""

    rows.append(
        {
            "section": section,
            "grouping": grouping,
            "group": group,
            "metric": metric,
            "value": value,
            "count": count,
            "percentage": percentage,
            "reddit_id": "",
            "conceptual_chunk_number": "",
            "whitespace_word_number": "",
            "original_word_character_length": "",
            "original_token_piece_count": "",
        }
    )


def descriptive_metrics(data: pd.DataFrame) -> dict[str, float | int]:
    """Return the required sentiment and probability statistics for one group."""

    score = data["sentiment_score"]
    return {
        "posts": len(data),
        "mean_sentiment_score": score.mean(),
        "std_sentiment_score": score.std(ddof=1),
        "min_sentiment_score": score.min(),
        "median_sentiment_score": score.median(),
        "max_sentiment_score": score.max(),
        "mean_positive_probability": data["post_positive_probability"].mean(),
        "mean_neutral_probability": data["post_neutral_probability"].mean(),
        "mean_negative_probability": data["post_negative_probability"].mean(),
    }


def build_sentiment_summary(
    data: pd.DataFrame,
    runtime: dict[str, str],
    input_sha256: str,
    output_sha256: str,
    chunk_statistics: dict[str, int],
    replacement_audit: pd.DataFrame,
) -> pd.DataFrame:
    """Build overall, runtime, chunk, and grouped post-level diagnostics."""

    rows: list[dict[str, Any]] = []
    overall = descriptive_metrics(data)
    overall_metrics: dict[str, Any] = {
        "total_posts": len(data),
        "successful_inference": len(data),
        "failed_inference": 0,
        **{key: value for key, value in overall.items() if key != "posts"},
        **chunk_statistics,
    }
    for metric, value in overall_metrics.items():
        add_summary_value(rows, "overall", "overall", "all", metric, value)

    metadata = {
        **runtime,
        "input_sha256": input_sha256,
        "output_sha256": output_sha256,
    }
    for metric, value in metadata.items():
        add_summary_value(rows, "runtime", "overall", "all", metric, value)

    chunk_counts = data["n_chunks"].value_counts().sort_index()
    for n_chunks, count in chunk_counts.items():
        add_summary_value(
            rows,
            "n_chunks_distribution",
            "n_chunks",
            str(int(n_chunks)),
            "posts",
            int(count),
            count=int(count),
            percentage=float(count / len(data) * 100),
        )

    grouped = data.copy()
    grouped["year"] = pd.to_datetime(grouped["date_utc"]).dt.year.astype(int)
    title_only = parse_boolean_series(grouped["title_only"], "title_only")
    grouped["text_type"] = np.where(title_only, "title_only", "title_plus_body")
    for grouping in ["year", "subreddit", "text_type"]:
        for group, subset in grouped.groupby(grouping, sort=True):
            for metric, value in descriptive_metrics(subset).items():
                add_summary_value(
                    rows,
                    "descriptive_statistics",
                    grouping,
                    str(group),
                    metric,
                    value,
                )
    for replacement in replacement_audit.itertuples(index=False):
        audit_row = {
            "section": "tokenizer_safeguard_replacement",
            "grouping": "conceptual_chunk",
            "group": f"{replacement.reddit_id}:{replacement.conceptual_chunk_number}",
            "metric": "impossible_whitespace_token_replaced_with_unk",
            "value": "[UNK]",
            "count": "",
            "percentage": "",
            "reddit_id": replacement.reddit_id,
            "conceptual_chunk_number": replacement.conceptual_chunk_number,
            "whitespace_word_number": replacement.whitespace_word_number,
            "original_word_character_length": (
                replacement.original_word_character_length
            ),
            "original_token_piece_count": replacement.original_token_piece_count,
        }
        rows.append(audit_row)
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def build_class_distribution(data: pd.DataFrame) -> pd.DataFrame:
    """Count descriptive FinBERT classes overall and within required groups."""

    grouped = data.copy()
    grouped["year"] = pd.to_datetime(grouped["date_utc"]).dt.year.astype(int)
    title_only = parse_boolean_series(grouped["title_only"], "title_only")
    grouped["text_type"] = np.where(title_only, "title_only", "title_plus_body")
    rows: list[dict[str, Any]] = []

    def add_group(grouping: str, group: str, subset: pd.DataFrame) -> None:
        counts = subset["sentiment_label"].value_counts()
        denominator = len(subset)
        for label in SENTIMENT_LABELS:
            count = int(counts.get(label, 0))
            rows.append(
                {
                    "grouping": grouping,
                    "group": group,
                    "sentiment_label": label,
                    "count": count,
                    "percentage": count / denominator * 100,
                }
            )

    add_group("overall", "all", grouped)
    for grouping in ["year", "subreddit", "text_type"]:
        for group, subset in grouped.groupby(grouping, sort=True):
            add_group(grouping, str(group), subset)
    return pd.DataFrame(
        rows,
        columns=["grouping", "group", "sentiment_label", "count", "percentage"],
    )


def take_ranked(data: pd.DataFrame, columns: list[str], ascending: list[bool]) -> pd.DataFrame:
    """Take a deterministic three-row ranking, breaking ties by Reddit ID."""

    ranked = data.sort_values(
        [*columns, "id"], ascending=[*ascending, True], kind="stable"
    )
    if len(ranked) < REVIEW_EXAMPLES_PER_GROUP:
        raise ValueError("A manual-review category has fewer than three posts.")
    return ranked.head(REVIEW_EXAMPLES_PER_GROUP).copy()


def build_review_sample(data: pd.DataFrame) -> pd.DataFrame:
    """Build the deterministic rank-based qualitative FinBERT review sample."""

    selections: list[pd.DataFrame] = []

    def add(label: str, chosen: pd.DataFrame) -> None:
        review = chosen.copy()
        review.insert(0, "review_group", label)
        selections.append(review.loc[:, REVIEW_COLUMNS])

    add("strongly_positive", take_ranked(data, ["sentiment_score"], [False]))
    near_neutral = take_ranked(
        data.assign(_absolute_score=data["sentiment_score"].abs()),
        ["_absolute_score"],
        [True],
    )
    add("near_neutral", near_neutral)
    near_neutral_ids = set(near_neutral["id"].astype(str))

    mildly_positive_pool = data.loc[
        data["sentiment_score"].gt(0) & ~data["id"].isin(near_neutral_ids)
    ]
    add(
        "mildly_positive",
        take_ranked(mildly_positive_pool, ["sentiment_score"], [True]),
    )
    mildly_negative_pool = data.loc[
        data["sentiment_score"].lt(0) & ~data["id"].isin(near_neutral_ids)
    ].assign(_absolute_score=lambda frame: frame["sentiment_score"].abs())
    add(
        "mildly_negative",
        take_ranked(mildly_negative_pool, ["_absolute_score"], [True]),
    )
    add("strongly_negative", take_ranked(data, ["sentiment_score"], [True]))
    add(
        "high_neutral_probability",
        take_ranked(data, ["post_neutral_probability"], [False]),
    )

    title_only = data.loc[parse_boolean_series(data["title_only"], "title_only")]
    ordered_title_only = title_only.sort_values(
        ["sentiment_score", "id"], kind="stable"
    )
    title_positions = np.linspace(
        0, len(ordered_title_only) - 1, REVIEW_EXAMPLES_PER_GROUP, dtype=int
    )
    add("title_only", ordered_title_only.iloc[title_positions].copy())
    add(
        "long_multi_chunk",
        take_ranked(data, ["n_chunks"], [False]),
    )
    safeguarded_posts = data.loc[
        parse_boolean_series(
            data["tokenizer_safeguard_applied"],
            "tokenizer_safeguard_applied",
        )
    ].sort_values("id", kind="stable")
    if safeguarded_posts.empty:
        raise ValueError("No tokenizer-safeguard post is available for review.")
    add("tokenizer_safeguard", safeguarded_posts)

    sample = pd.concat(selections, ignore_index=True)
    sample["review_group"] = pd.Categorical(
        sample["review_group"], categories=REVIEW_GROUPS, ordered=True
    )
    sample = sample.sort_values(
        ["review_group", "id"], kind="stable"
    ).reset_index(drop=True)
    sample["review_group"] = sample["review_group"].astype(str)
    validate_review_sample(sample)
    return sample


def validate_review_sample(sample: pd.DataFrame) -> None:
    """Validate the fixed review categories and schema."""

    if list(sample.columns) != REVIEW_COLUMNS:
        raise ValueError("FinBERT review sample schema is incorrect.")
    counts = sample["review_group"].value_counts()
    if set(counts.index) != set(REVIEW_GROUPS):
        raise ValueError("FinBERT review sample lacks a required review group.")
    regular_groups = [
        group for group in REVIEW_GROUPS if group != "tokenizer_safeguard"
    ]
    if not counts.reindex(regular_groups).eq(REVIEW_EXAMPLES_PER_GROUP).all():
        raise ValueError("Every regular FinBERT review group must contain three examples.")
    safeguard_rows = sample.loc[
        sample["review_group"].eq("tokenizer_safeguard")
    ]
    if safeguard_rows.empty or not parse_boolean_series(
        safeguard_rows["tokenizer_safeguard_applied"],
        "tokenizer_safeguard_applied",
    ).all():
        raise ValueError("Tokenizer-safeguard review rows are missing or invalid.")


def validate_diagnostics(
    summary: pd.DataFrame,
    distribution: pd.DataFrame,
    review: pd.DataFrame,
    input_sha256: str,
    output_rows: int,
) -> None:
    """Validate required diagnostic structure and count reconciliations."""

    if set(summary.columns) != set(SUMMARY_COLUMNS):
        raise ValueError("FinBERT sentiment summary schema is incorrect.")
    input_hash_rows = summary.loc[summary["metric"].eq("input_sha256"), "value"]
    if len(input_hash_rows) != 1 or str(input_hash_rows.iloc[0]) != input_sha256:
        raise ValueError("FinBERT summary does not identify the current Phase 3B input.")

    required_distribution_columns = {
        "grouping", "group", "sentiment_label", "count", "percentage"
    }
    if set(distribution.columns) != required_distribution_columns:
        raise ValueError("FinBERT class distribution schema is incorrect.")
    overall = distribution.loc[distribution["grouping"].eq("overall")]
    if set(overall["sentiment_label"]) != set(SENTIMENT_LABELS):
        raise ValueError("Overall class distribution lacks a FinBERT label.")
    if int(pd.to_numeric(overall["count"]).sum()) != output_rows:
        raise ValueError("Overall class counts do not reconcile to output rows.")
    validate_review_sample(review)


def write_phase4a_outputs(
    output: pd.DataFrame,
    class_distribution: pd.DataFrame,
    review_sample: pd.DataFrame,
    runtime: dict[str, str],
    input_sha256: str,
    chunk_statistics: dict[str, int],
    replacement_audit: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Write validated Phase 4A files and return their final checksums."""

    for path in [
        FINBERT_REDDIT_FILE,
        FINBERT_SENTIMENT_SUMMARY_FILE,
        FINBERT_CLASS_DISTRIBUTION_FILE,
        FINBERT_REVIEW_SAMPLE_FILE,
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)

    output.to_csv(FINBERT_REDDIT_FILE, index=False, encoding="utf-8-sig")
    output_sha256 = file_sha256(FINBERT_REDDIT_FILE)
    summary = build_sentiment_summary(
        output,
        runtime,
        input_sha256,
        output_sha256,
        chunk_statistics,
        replacement_audit,
    )
    validate_diagnostics(
        summary, class_distribution, review_sample, input_sha256, len(output)
    )
    summary.to_csv(
        FINBERT_SENTIMENT_SUMMARY_FILE, index=False, encoding="utf-8-sig"
    )
    class_distribution.to_csv(
        FINBERT_CLASS_DISTRIBUTION_FILE, index=False, encoding="utf-8-sig"
    )
    review_sample.to_csv(
        FINBERT_REVIEW_SAMPLE_FILE, index=False, encoding="utf-8-sig"
    )
    checksums = {
        str(path): file_sha256(path)
        for path in [
            FINBERT_REDDIT_FILE,
            FINBERT_SENTIMENT_SUMMARY_FILE,
            FINBERT_CLASS_DISTRIBUTION_FILE,
            FINBERT_REVIEW_SAMPLE_FILE,
        ]
    }
    return summary, checksums


def existing_outputs_are_valid(input_data: pd.DataFrame, input_sha256: str) -> bool:
    """Validate existing Phase 4A files so expensive inference can be skipped."""

    paths = [
        FINBERT_REDDIT_FILE,
        FINBERT_SENTIMENT_SUMMARY_FILE,
        FINBERT_CLASS_DISTRIBUTION_FILE,
        FINBERT_REVIEW_SAMPLE_FILE,
    ]
    if not all(path.exists() for path in paths):
        return False
    try:
        output = pd.read_csv(
            FINBERT_REDDIT_FILE,
            dtype={"id": "string", "subreddit": "string"},
            keep_default_na=False,
        )
        validate_post_output(input_data, output)
        summary = pd.read_csv(FINBERT_SENTIMENT_SUMMARY_FILE, keep_default_na=False)
        distribution = pd.read_csv(
            FINBERT_CLASS_DISTRIBUTION_FILE, keep_default_na=False
        )
        review = pd.read_csv(
            FINBERT_REVIEW_SAMPLE_FILE,
            dtype={"id": "string", "subreddit": "string"},
            keep_default_na=False,
        )
        validate_diagnostics(
            summary, distribution, review, input_sha256, len(output)
        )
        recorded_output_hash = summary.loc[
            summary["metric"].eq("output_sha256"), "value"
        ]
        if len(recorded_output_hash) != 1 or str(recorded_output_hash.iloc[0]) != file_sha256(
            FINBERT_REDDIT_FILE
        ):
            raise ValueError("Recorded Phase 4A output checksum is stale.")
    except Exception as error:
        print(f"Existing Phase 4A outputs are incomplete or invalid: {error}")
        return False
    return True


def print_output_report(
    data: pd.DataFrame,
    checksums: dict[str, str],
    chunk_statistics: dict[str, int],
) -> None:
    """Print concise validated results and permanent-file checksums."""

    probabilities = data[PROBABILITY_COLUMNS]
    print(f"Output posts: {len(data):,}")
    print(f"Unique IDs: {data['id'].nunique():,}")
    print(f"Duplicate IDs: {int(data['id'].duplicated().sum()):,}")
    print("Failed inference: 0")
    print(f"Total uncapped chunks: {chunk_statistics['total_uncapped_chunks']:,}")
    print(f"Total scored chunks: {chunk_statistics['total_scored_chunks']:,}")
    print(
        "Maximum pre-cap chunks for one post: "
        f"{chunk_statistics['maximum_uncapped_chunks_per_post']:,}"
    )
    print(
        "Tokenizer safeguard (oversized/[UNK]/fragmented/final inputs/max tokens/"
        "still exceeding): "
        f"{chunk_statistics['oversized_retained_conceptual_chunks_before_safeguard']}/"
        f"{chunk_statistics['impossible_whitespace_tokens_replaced_with_unk']}/"
        f"{chunk_statistics['conceptual_chunks_requiring_internal_fragmentation']}/"
        f"{chunk_statistics['total_final_model_inputs']}/"
        f"{chunk_statistics['maximum_final_model_input_token_length']}/"
        f"{chunk_statistics['final_model_inputs_exceeding_limit']}"
    )
    print(
        "Sentiment score range: "
        f"[{data['sentiment_score'].min():.6f}, "
        f"{data['sentiment_score'].max():.6f}]"
    )
    print(
        "n_chunks min/median/mean/max: "
        f"{data['n_chunks'].min()}/"
        f"{data['n_chunks'].median():.1f}/"
        f"{data['n_chunks'].mean():.3f}/"
        f"{data['n_chunks'].max()}"
    )
    print(
        "Posts hitting chunk cap: "
        f"{chunk_statistics['posts_hitting_120_chunk_cap']:,}"
    )
    print(
        "Mean probabilities (positive/neutral/negative): "
        f"{probabilities.mean().iloc[0]:.6f}/"
        f"{probabilities.mean().iloc[1]:.6f}/"
        f"{probabilities.mean().iloc[2]:.6f}"
    )
    print("Phase 4A checksums:")
    for path, checksum in checksums.items():
        print(f"  {path}: {checksum}")


def main() -> None:
    """Run Phase 4A only: post-level FinBERT probability inference."""

    arguments = parse_arguments()
    validate_configuration()
    if arguments.force and arguments.resume:
        raise ValueError("--force and --resume cannot be used together.")
    if arguments.batch_size < 1 or arguments.checkpoint_interval < 1:
        raise ValueError("Batch size and checkpoint interval must be positive.")
    input_data, input_sha256 = load_and_validate_input()
    print(f"Validated input SHA-256: {input_sha256}")
    print(
        f"Input rows={len(input_data):,}, unique IDs={input_data['id'].nunique():,}, "
        f"duplicate IDs={int(input_data['id'].duplicated().sum()):,}, "
        "blank finbert_text=0."
    )
    if arguments.validate_input_only:
        print("Input-only validation completed; FinBERT was not loaded.")
        return

    if not arguments.force and existing_outputs_are_valid(input_data, input_sha256):
        print("Valid complete Phase 4A outputs already exist; inference was skipped.")
        return

    # Conceptual chunks define the weighting unit within each post. Tokenizer
    # safeguards below may change the number of model calls, never these weights.
    uncapped_chunks, post_metadata = build_chunk_table(input_data)
    chunk_statistics = {
        "total_uncapped_chunks": len(uncapped_chunks),
        "total_scored_chunks": int(post_metadata["n_chunks"].sum()),
        "maximum_uncapped_chunks_per_post": int(
            post_metadata["uncapped_n_chunks"].max()
        ),
        "posts_hitting_120_chunk_cap": int(
            post_metadata["chunk_cap_applied"].sum()
        ),
    }
    print(
        f"Built {len(uncapped_chunks):,} uncapped chunks for "
        f"{len(input_data):,} posts; maximum pre-cap count="
        f"{chunk_statistics['maximum_uncapped_chunks_per_post']:,}."
    )
    tokenizer, model, label_mapping, runtime, torch = load_finbert()
    conceptual_chunks = apply_chunk_cap(uncapped_chunks, post_metadata)
    model_inputs, replacement_audit, tokenizer_statistics = prepare_model_inputs(
        conceptual_chunks, tokenizer, model
    )
    chunk_statistics.update(tokenizer_statistics)
    if len(conceptual_chunks) != EXPECTED_RETAINED_CONCEPTUAL_CHUNKS:
        raise ValueError(
            "Retained conceptual chunks do not reconcile to the validated 34,879."
        )
    if len(model_inputs) != EXPECTED_MODEL_INPUTS:
        raise ValueError("Final model inputs do not reconcile to the validated 34,884.")
    if (
        tokenizer_statistics["conceptual_chunks_requiring_internal_fragmentation"]
        != EXPECTED_FRAGMENTED_CONCEPTUAL_CHUNKS
        or tokenizer_statistics["total_internal_model_fragments"]
        != EXPECTED_INTERNAL_MODEL_FRAGMENTS
        or tokenizer_statistics[
            "additional_model_inputs_from_internal_fragmentation"
        ]
        != EXPECTED_MODEL_INPUTS - EXPECTED_RETAINED_CONCEPTUAL_CHUNKS
    ):
        raise ValueError("Internal model-fragment reconciliation is incorrect.")
    print(
        f"Loaded {runtime['model_identifier']} on {runtime['inference_device']}; "
        "validated every retained chunk that will be sent to FinBERT."
    )
    print(
        f"After the cap, {len(conceptual_chunks):,} conceptual chunks will be "
        f"represented by {len(model_inputs):,} model inputs; "
        f"{chunk_statistics['posts_hitting_120_chunk_cap']:,} posts were capped."
    )
    processing_order, token_lengths = build_length_aware_processing_order(
        model_inputs, tokenizer
    )
    fingerprint = model_input_fingerprint(
        model_inputs,
        input_sha256,
        runtime["model_identifier"],
        runtime["model_revision"],
    )
    metadata = {
        "input_sha256": input_sha256,
        "model_identifier": runtime["model_identifier"],
        "model_revision": runtime["model_revision"],
        "transformers_version": runtime["transformers_version"],
        "torch_version": runtime["torch_version"],
        "batch_size": arguments.batch_size,
        "checkpoint_interval": arguments.checkpoint_interval,
        "retained_conceptual_chunks": len(conceptual_chunks),
        "model_inputs": len(model_inputs),
        "fragmented_conceptual_chunks": tokenizer_statistics[
            "conceptual_chunks_requiring_internal_fragmentation"
        ],
        "internal_model_fragments": tokenizer_statistics[
            "total_internal_model_fragments"
        ],
        "model_input_fingerprint": fingerprint,
        "processing_order_sha256": hashlib.sha256(
            ",".join(str(position) for position in processing_order).encode("ascii")
        ).hexdigest().upper(),
        "maximum_model_input_tokens": max(token_lengths),
    }
    probability_rows = initialize_or_validate_checkpoint(
        model_inputs,
        processing_order,
        metadata,
        arguments.resume,
    )
    print(
        f"Production inference | batch_size={arguments.batch_size} | "
        f"checkpoint_interval={arguments.checkpoint_interval} | "
        f"resuming_from={len(probability_rows):,} | "
        f"torch_threads={torch.get_num_threads()} | "
        f"torch_interop_threads={torch.get_num_interop_threads()}",
        flush=True,
    )
    scored_model_inputs = score_model_inputs(
        model_inputs,
        tokenizer,
        model,
        label_mapping,
        torch,
        processing_order,
        batch_size=arguments.batch_size,
        checkpoint_interval=arguments.checkpoint_interval,
        probability_rows=probability_rows,
    )
    scored_chunks = reconstruct_conceptual_probabilities(
        scored_model_inputs, conceptual_chunks
    )
    if len(scored_chunks) != EXPECTED_RETAINED_CONCEPTUAL_CHUNKS:
        raise ValueError("Model fragments did not reconstruct exactly 34,879 chunks.")
    output = aggregate_posts(input_data, scored_chunks, post_metadata)
    validate_post_output(input_data, output)
    class_distribution = build_class_distribution(output)
    review_sample = build_review_sample(output)
    _, checksums = write_phase4a_outputs(
        output,
        class_distribution,
        review_sample,
        runtime,
        input_sha256,
        chunk_statistics,
        replacement_audit,
    )
    print_output_report(output, checksums, chunk_statistics)
    print("Phase 4A post-level FinBERT inference completed successfully.")


if __name__ == "__main__":
    main()
