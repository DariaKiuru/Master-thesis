"""Development runner for checkpointed batched Phase 4A FinBERT inference.

This file is deliberately outside the active ``src`` pipeline while the
validated Phase 4A worker is running. It reuses the production chunking,
tokenizer safeguards, probability reconstruction, and post aggregation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from config import (  # noqa: E402
    FINBERT_BATCH_SIZE,
    FINBERT_BENCHMARK_FILE,
    FINBERT_CHECKPOINT_DIR,
    FINBERT_DEVELOPMENT_SAMPLE_FILE,
)


CHECKPOINT_INTERVAL = 800
DIAGNOSTIC_SAMPLE_SIZE = 45
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
DIAGNOSTIC_OUTPUT_COLUMNS = [
    "id",
    "date_utc",
    "subreddit",
    "n_chunks",
    "post_positive_probability",
    "post_neutral_probability",
    "post_negative_probability",
    "sentiment_score",
    "sentiment_label",
    "tokenizer_safeguard_applied",
]


def load_production_module() -> Any:
    """Load the active Phase 4A functions without duplicating methodology."""

    script_path = REPOSITORY_ROOT / "src" / "07_score_finbert.py"
    specification = importlib.util.spec_from_file_location(
        "phase4a_production", script_path
    )
    if specification is None or specification.loader is None:
        raise ImportError(f"Could not load production Phase 4A script: {script_path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def parse_arguments() -> argparse.Namespace:
    """Require an explicit development mode and expose computational controls."""

    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--diagnostic-sample",
        action="store_true",
        help="Score a deterministic 45-post sample, never the final empirical file.",
    )
    mode.add_argument(
        "--full",
        action="store_true",
        help="Run the full checkpointed Phase 4A workflow when no other worker exists.",
    )
    mode.add_argument(
        "--benchmark",
        action="store_true",
        help="Benchmark representative model inputs; never write empirical scores.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--batch-size", type=int, default=FINBERT_BATCH_SIZE)
    parser.add_argument("--checkpoint-interval", type=int, default=CHECKPOINT_INTERVAL)
    parser.add_argument(
        "--length-aware-batching",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Group similar token lengths while restoring logical order afterward.",
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        help="Optionally set PyTorch CPU intra-op threads; defaults remain unchanged.",
    )
    return parser.parse_args()


def deterministic_diagnostic_sample(data: pd.DataFrame) -> pd.DataFrame:
    """Select 45 posts evenly across the nine year/subreddit strata."""

    working = data.copy()
    working["_year"] = pd.to_datetime(working["date_utc"]).dt.year.astype(int)
    selected_indices: set[int] = set()
    for _, group in working.groupby(["_year", "subreddit"], sort=True):
        ordered = group.sort_values(["date_utc", "id"], kind="stable")
        take = min(5, len(ordered))
        positions = np.linspace(0, len(ordered) - 1, take, dtype=int)
        selected_indices.update(ordered.iloc[positions].index.tolist())

    if len(selected_indices) < DIAGNOSTIC_SAMPLE_SIZE:
        remaining = working.loc[~working.index.isin(selected_indices)].sort_values(
            ["date_utc", "subreddit", "id"], kind="stable"
        )
        needed = DIAGNOSTIC_SAMPLE_SIZE - len(selected_indices)
        selected_indices.update(remaining.head(needed).index.tolist())
    sample = working.loc[sorted(selected_indices)].drop(columns="_year")
    sample = sample.head(DIAGNOSTIC_SAMPLE_SIZE).reset_index(drop=True)
    if len(sample) != DIAGNOSTIC_SAMPLE_SIZE or sample["id"].duplicated().any():
        raise ValueError("The diagnostic FinBERT sample must contain 45 unique posts.")
    represented = set(
        zip(pd.to_datetime(sample["date_utc"]).dt.year, sample["subreddit"])
    )
    if len(represented) != 9:
        raise ValueError("The diagnostic sample does not cover every year/subreddit stratum.")
    return sample


def model_input_fingerprint(
    model_inputs: pd.DataFrame,
    input_sha256: str,
    model_identifier: str,
    model_revision: str,
    mode: str,
) -> str:
    """Identify the exact ordered model-input sequence behind a checkpoint."""

    digest = hashlib.sha256()
    digest.update(
        f"{input_sha256}|{model_identifier}|{model_revision}|{mode}|"
        f"{len(model_inputs)}".encode("utf-8")
    )
    for row in model_inputs.itertuples(index=False):
        digest.update(
            f"\n{row.post_position}|{row.id}|{row.chunk_number}|"
            f"{row.fragment_number}|{row.model_input_text}".encode("utf-8")
        )
    return digest.hexdigest().upper()


def build_processing_order(
    model_inputs: pd.DataFrame,
    tokenizer: Any,
    length_aware: bool,
) -> tuple[list[int], list[int]]:
    """Return a deterministic inference order and exact final token lengths."""

    encoded = tokenizer(
        model_inputs["model_input_text"].tolist(),
        add_special_tokens=True,
        truncation=False,
        padding=False,
    )["input_ids"]
    token_lengths = [len(input_ids) for input_ids in encoded]
    if length_aware:
        processing_order = sorted(
            range(len(model_inputs)),
            key=lambda position: (token_lengths[position], position),
        )
    else:
        processing_order = list(range(len(model_inputs)))
    if sorted(processing_order) != list(range(len(model_inputs))):
        raise ValueError("Inference processing order is not a full permutation.")
    return processing_order, token_lengths


def checkpoint_paths(mode: str) -> tuple[Path, Path]:
    """Return ignored checkpoint data and metadata paths for one run mode."""

    return (
        FINBERT_CHECKPOINT_DIR / f"{mode}_probabilities.csv",
        FINBERT_CHECKPOINT_DIR / f"{mode}_metadata.json",
    )


def write_checkpoint_atomic(
    model_inputs: pd.DataFrame,
    processing_order: list[int],
    probability_rows: list[dict[str, float]],
    checkpoint_file: Path,
) -> None:
    """Atomically persist the exact completed prefix and its probabilities."""

    completed = len(probability_rows)
    completed_positions = processing_order[:completed]
    checkpoint = model_inputs.iloc[completed_positions].loc[
        :, CHECKPOINT_KEY_COLUMNS
    ].reset_index(drop=True)
    checkpoint.insert(0, "logical_position", completed_positions)
    probabilities = pd.DataFrame(
        probability_rows, columns=CHECKPOINT_PROBABILITY_COLUMNS
    )
    checkpoint = pd.concat(
        [checkpoint.reset_index(drop=True), probabilities.reset_index(drop=True)],
        axis="columns",
    )
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = checkpoint_file.with_suffix(".tmp")
    checkpoint.to_csv(temporary_file, index=False)
    temporary_file.replace(checkpoint_file)


def initialize_or_validate_checkpoint(
    model_inputs: pd.DataFrame,
    processing_order: list[int],
    checkpoint_file: Path,
    metadata_file: Path,
    metadata: dict[str, Any],
    resume: bool,
    production: Any,
) -> list[dict[str, float]]:
    """Load only a validated contiguous prefix or initialize an empty run."""

    if not resume:
        if checkpoint_file.exists() or metadata_file.exists():
            raise FileExistsError(
                "A development checkpoint already exists. Use --resume or move it."
            )
        metadata_file.parent.mkdir(parents=True, exist_ok=True)
        temporary_metadata = metadata_file.with_suffix(".tmp")
        temporary_metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        temporary_metadata.replace(metadata_file)
        return []

    if not checkpoint_file.exists() or not metadata_file.exists():
        raise FileNotFoundError("Both checkpoint data and metadata are required to resume.")
    recorded_metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    if recorded_metadata != metadata:
        raise ValueError("Checkpoint metadata does not match the current model inputs.")
    checkpoint = pd.read_csv(checkpoint_file, dtype={"id": "string"})
    expected_columns = [*CHECKPOINT_COLUMNS, *CHECKPOINT_PROBABILITY_COLUMNS]
    if list(checkpoint.columns) != expected_columns:
        raise ValueError("Checkpoint schema is invalid.")
    if len(checkpoint) > len(model_inputs):
        raise ValueError("Checkpoint contains more rows than the current model inputs.")
    expected_positions = processing_order[: len(checkpoint)]
    recorded_positions = pd.to_numeric(
        checkpoint["logical_position"], errors="coerce"
    )
    if recorded_positions.isna().any() or not np.array_equal(
        recorded_positions.to_numpy(dtype=int), expected_positions
    ):
        raise ValueError("Checkpoint processing order differs from the current run.")
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
        raise ValueError("Checkpoint is not an exact contiguous model-input prefix.")
    production.validate_chunk_probabilities(checkpoint)
    return checkpoint.loc[:, CHECKPOINT_PROBABILITY_COLUMNS].to_dict("records")


def score_model_inputs_checkpointed(
    model_inputs: pd.DataFrame,
    tokenizer: Any,
    model: Any,
    label_mapping: dict[int, str],
    torch: Any,
    *,
    batch_size: int,
    checkpoint_interval: int,
    checkpoint_file: Path,
    processing_order: list[int],
    probability_rows: list[dict[str, float]],
    production: Any,
) -> pd.DataFrame:
    """Score ordered model inputs in batches with recoverable prefix checkpoints."""

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
        if probabilities.shape != (len(batch), len(label_mapping)):
            raise ValueError(f"Unexpected probability shape: {probabilities.shape}.")
        if not np.isfinite(probabilities).all() or not np.allclose(
            probabilities.sum(axis=1), 1.0, atol=1e-5, rtol=0
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
            write_checkpoint_atomic(
                model_inputs,
                processing_order,
                probability_rows,
                checkpoint_file,
            )
            elapsed = time.perf_counter() - started
            newly_scored = completed - completed_at_start
            throughput = newly_scored / elapsed if elapsed else float("nan")
            print(
                f"Checkpoint | completed={completed:,}/{len(model_inputs):,} | "
                f"elapsed={elapsed:.1f}s | throughput={throughput:.2f} inputs/s | "
                f"file={checkpoint_file}",
                flush=True,
            )
            while next_checkpoint <= completed:
                next_checkpoint += checkpoint_interval

    if len(probability_rows) != len(model_inputs):
        raise ValueError("Checkpointed inference did not score every model input.")
    ordered_probability_rows: list[dict[str, float] | None] = [
        None
    ] * len(model_inputs)
    for logical_position, probabilities in zip(processing_order, probability_rows):
        ordered_probability_rows[logical_position] = probabilities
    if any(row is None for row in ordered_probability_rows):
        raise ValueError("Length-aware batching did not restore every logical position.")
    scored = model_inputs.copy()
    probabilities = pd.DataFrame(
        ordered_probability_rows, columns=CHECKPOINT_PROBABILITY_COLUMNS
    )
    for column in CHECKPOINT_PROBABILITY_COLUMNS:
        scored[column] = probabilities[column].to_numpy()
    production.validate_chunk_probabilities(scored)
    return scored


def representative_benchmark_positions(
    model_inputs: pd.DataFrame,
    token_lengths: list[int],
    requested_size: int = 960,
) -> list[int]:
    """Sample the empirical token-length distribution deterministically."""

    length_order = sorted(
        range(len(model_inputs)),
        key=lambda position: (token_lengths[position], position),
    )
    sample_size = min(requested_size, len(length_order))
    quantile_positions = np.linspace(
        0, len(length_order) - 1, sample_size, dtype=int
    )
    selected = {length_order[position] for position in quantile_positions}
    safeguarded = model_inputs.index[
        model_inputs["tokenizer_safeguard_applied"].astype(bool)
    ].tolist()
    selected.update(int(position) for position in safeguarded)
    return sorted(selected)


def benchmark_configuration(
    model_inputs: pd.DataFrame,
    tokenizer: Any,
    model: Any,
    torch: Any,
    token_lengths: list[int],
    sample_positions: list[int],
    *,
    batch_size: int,
    length_aware: bool,
    full_model_input_count: int,
) -> dict[str, Any]:
    """Benchmark one computational configuration without retaining scores."""

    if length_aware:
        order = sorted(
            sample_positions,
            key=lambda position: (token_lengths[position], position),
        )
    else:
        order = sorted(sample_positions)
    if not order:
        raise ValueError("Benchmark sample is empty.")

    warmup_positions = order[:batch_size]
    warmup = tokenizer(
        model_inputs.iloc[warmup_positions]["model_input_text"].tolist(),
        padding=True,
        truncation=False,
        return_tensors="pt",
    )
    device = next(model.parameters()).device
    warmup = {name: values.to(device) for name, values in warmup.items()}
    with torch.inference_mode():
        model(**warmup)

    padded_token_slots = 0
    actual_token_slots = 0
    started = time.perf_counter()
    for start in range(0, len(order), batch_size):
        positions = order[start : start + batch_size]
        texts = model_inputs.iloc[positions]["model_input_text"].tolist()
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=False,
            return_tensors="pt",
        )
        padded_token_slots += int(encoded["input_ids"].numel())
        actual_token_slots += sum(token_lengths[position] for position in positions)
        encoded = {name: values.to(device) for name, values in encoded.items()}
        with torch.inference_mode():
            logits = model(**encoded).logits
            probabilities = torch.softmax(logits, dim=-1)
        if probabilities.shape[0] != len(positions):
            raise ValueError("Benchmark inference returned an unexpected row count.")
    elapsed = time.perf_counter() - started
    throughput = len(order) / elapsed
    estimated_seconds = full_model_input_count / throughput
    padding_fraction = (
        (padded_token_slots - actual_token_slots) / padded_token_slots
        if padded_token_slots
        else float("nan")
    )
    return {
        "batch_size": batch_size,
        "length_aware_batching": length_aware,
        "sample_model_inputs": len(order),
        "sample_mean_token_length": float(
            np.mean([token_lengths[position] for position in order])
        ),
        "sample_max_token_length": max(token_lengths[position] for position in order),
        "padding_fraction": padding_fraction,
        "elapsed_seconds": elapsed,
        "model_inputs_per_second": throughput,
        "estimated_full_run_seconds": estimated_seconds,
        "estimated_full_run_minutes": estimated_seconds / 60,
        "full_model_input_count": full_model_input_count,
        "torch_cpu_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "inference_context": "torch.inference_mode",
    }


def run_benchmarks(
    model_inputs: pd.DataFrame,
    tokenizer: Any,
    model: Any,
    torch: Any,
    token_lengths: list[int],
) -> pd.DataFrame:
    """Compare current and length-aware batch sizes on one fixed sample."""

    sample_positions = representative_benchmark_positions(
        model_inputs, token_lengths
    )
    configurations = [
        (FINBERT_BATCH_SIZE, False),
        (8, True),
        (16, True),
        (32, True),
    ]
    rows = []
    for batch_size, length_aware in configurations:
        result = benchmark_configuration(
            model_inputs,
            tokenizer,
            model,
            torch,
            token_lengths,
            sample_positions,
            batch_size=batch_size,
            length_aware=length_aware,
            full_model_input_count=len(model_inputs),
        )
        rows.append(result)
        print(
            f"Benchmark | batch={batch_size} | length_aware={length_aware} | "
            f"throughput={result['model_inputs_per_second']:.2f} inputs/s | "
            f"estimated_full={result['estimated_full_run_minutes']:.1f} min | "
            f"padding={result['padding_fraction']:.2%}",
            flush=True,
        )
    benchmark = pd.DataFrame(rows)
    FINBERT_BENCHMARK_FILE.parent.mkdir(parents=True, exist_ok=True)
    benchmark.to_csv(FINBERT_BENCHMARK_FILE, index=False)
    print(f"Saved development benchmark to {FINBERT_BENCHMARK_FILE}")
    return benchmark


def validate_diagnostic_output(data: pd.DataFrame, expected_rows: int) -> None:
    """Validate diagnostic post probabilities without treating them as final data."""

    if len(data) != expected_rows or data["id"].duplicated().any():
        raise ValueError("Diagnostic output changed sample membership.")
    probabilities = data[production_probability_columns()].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all():
        raise ValueError("Diagnostic probabilities contain a non-finite value.")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5, rtol=0):
        raise ValueError("Diagnostic probabilities do not sum to one.")
    expected_scores = probabilities[:, 0] - probabilities[:, 2]
    if not np.allclose(data["sentiment_score"], expected_scores, atol=1e-10, rtol=0):
        raise ValueError("Diagnostic sentiment scores are inconsistent.")
    expected_labels = np.asarray(["positive", "neutral", "negative"])[
        probabilities.argmax(axis=1)
    ]
    if not np.array_equal(data["sentiment_label"].astype(str), expected_labels):
        raise ValueError("Diagnostic labels do not equal probability argmax.")


def production_probability_columns() -> list[str]:
    """Return the locked post-level probability column order."""

    return [
        "post_positive_probability",
        "post_neutral_probability",
        "post_negative_probability",
    ]


def main() -> None:
    """Run an explicit diagnostic or future full checkpointed development mode."""

    arguments = parse_arguments()
    production = load_production_module()
    production.validate_configuration()
    full_input, input_sha256 = production.load_and_validate_input()
    if arguments.diagnostic_sample:
        mode = "diagnostic"
    elif arguments.benchmark:
        mode = "benchmark"
    else:
        mode = "full"
    posts = (
        deterministic_diagnostic_sample(full_input)
        if arguments.diagnostic_sample
        else full_input
    )
    posts = posts.reset_index(drop=True)
    uncapped_chunks, post_metadata = production.build_chunk_table(posts)
    conceptual_chunks = production.apply_chunk_cap(uncapped_chunks, post_metadata)
    tokenizer, model, label_mapping, runtime, torch = production.load_finbert()
    if arguments.cpu_threads is not None:
        if arguments.cpu_threads < 1:
            raise ValueError("--cpu-threads must be positive.")
        torch.set_num_threads(arguments.cpu_threads)
    print(
        f"Runtime | device={runtime['inference_device']} | batch_size="
        f"{arguments.batch_size} | torch_threads={torch.get_num_threads()} | "
        f"torch_interop_threads={torch.get_num_interop_threads()}",
        flush=True,
    )
    model_inputs, replacement_audit, tokenizer_statistics = (
        production.prepare_model_inputs(conceptual_chunks, tokenizer, model)
    )
    processing_order, token_lengths = build_processing_order(
        model_inputs,
        tokenizer,
        arguments.length_aware_batching,
    )
    if arguments.benchmark:
        run_benchmarks(model_inputs, tokenizer, model, torch, token_lengths)
        return
    fingerprint = model_input_fingerprint(
        model_inputs,
        input_sha256,
        runtime["model_identifier"],
        runtime["model_revision"],
        mode,
    )
    checkpoint_file, metadata_file = checkpoint_paths(mode)
    metadata = {
        "mode": mode,
        "input_sha256": input_sha256,
        "model_identifier": runtime["model_identifier"],
        "model_revision": runtime["model_revision"],
        "post_count": len(posts),
        "model_input_count": len(model_inputs),
        "model_input_fingerprint": fingerprint,
        "length_aware_batching": arguments.length_aware_batching,
        "processing_order_sha256": hashlib.sha256(
            ",".join(str(position) for position in processing_order).encode("ascii")
        ).hexdigest().upper(),
    }
    probability_rows = initialize_or_validate_checkpoint(
        model_inputs,
        processing_order,
        checkpoint_file,
        metadata_file,
        metadata,
        arguments.resume,
        production,
    )
    print(f"Resuming from {len(probability_rows):,} completed model inputs.")
    scored_model_inputs = score_model_inputs_checkpointed(
        model_inputs,
        tokenizer,
        model,
        label_mapping,
        torch,
        batch_size=arguments.batch_size,
        checkpoint_interval=arguments.checkpoint_interval,
        checkpoint_file=checkpoint_file,
        processing_order=processing_order,
        probability_rows=probability_rows,
        production=production,
    )
    scored_chunks = production.reconstruct_conceptual_probabilities(
        scored_model_inputs, conceptual_chunks
    )
    output = production.aggregate_posts(posts, scored_chunks, post_metadata)

    if arguments.diagnostic_sample:
        validate_diagnostic_output(output, len(posts))
        diagnostic = output.loc[:, DIAGNOSTIC_OUTPUT_COLUMNS]
        FINBERT_DEVELOPMENT_SAMPLE_FILE.parent.mkdir(parents=True, exist_ok=True)
        diagnostic.to_csv(
            FINBERT_DEVELOPMENT_SAMPLE_FILE, index=False, encoding="utf-8-sig"
        )
        print(
            "Development diagnostic only; not final empirical data. Saved to "
            f"{FINBERT_DEVELOPMENT_SAMPLE_FILE}"
        )
        return

    production.validate_post_output(full_input, output)
    chunk_statistics = {
        "total_uncapped_chunks": len(uncapped_chunks),
        "total_scored_chunks": int(post_metadata["n_chunks"].sum()),
        "maximum_uncapped_chunks_per_post": int(
            post_metadata["uncapped_n_chunks"].max()
        ),
        "posts_hitting_120_chunk_cap": int(
            post_metadata["chunk_cap_applied"].sum()
        ),
        **tokenizer_statistics,
    }
    class_distribution = production.build_class_distribution(output)
    review_sample = production.build_review_sample(output)
    _, checksums = production.write_phase4a_outputs(
        output,
        class_distribution,
        review_sample,
        runtime,
        input_sha256,
        chunk_statistics,
        replacement_audit,
    )
    production.print_output_report(output, checksums, chunk_statistics)


if __name__ == "__main__":
    main()
