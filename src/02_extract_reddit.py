"""Extract and validate the Phase 3A raw candidate Reddit corpus."""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests


# Allow ``python src/02_extract_reddit.py`` from the repository root.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from config import (  # noqa: E402
    END_DATE,
    RAW_REDDIT_FILE,
    REDDIT_CHECKPOINT_POSTS_FILE,
    REDDIT_CHECKPOINT_WINDOWS_FILE,
    REDDIT_EXTRACTION_KEYWORDS,
    REDDIT_EXTRACTION_SUMMARY_FILE,
    REDDIT_QUERY_SUMMARY_FILE,
    START_DATE,
    SUBREDDITS,
)


ARCTIC_SHIFT_POSTS_URL = (
    "https://arctic-shift.photon-reddit.com/api/posts/search"
)
ARCTIC_SHIFT_SOURCE_NAME = "Arctic Shift historical Reddit posts API"
# A combined OR query was rejected empirically (0 IDs versus 167 for the
# individual-query union). Production extraction uses individual terms only.

REQUEST_FIELDS = [
    "id",
    "created_utc",
    "subreddit",
    "title",
    "selftext",
    "score",
    "num_comments",
    "url",
]
CHECKPOINT_POST_COLUMNS = [*REQUEST_FIELDS, "matched_search_keywords"]
RAW_OUTPUT_COLUMNS = [
    *REQUEST_FIELDS,
    "created_datetime_utc",
    "date_utc",
    "matched_search_keywords",
    "extraction_source",
    "source_endpoint",
    "extracted_at_utc",
]
CHECKPOINT_WINDOW_COLUMNS = [
    "recorded_at_utc",
    "subreddit",
    "initial_window",
    "final_window",
    "window_start_utc",
    "window_end_utc_exclusive",
    "query_used",
    "result_count",
    "api_request_count",
    "retry_count",
    "timeout_422_count",
    "rate_limit_429_count",
    "split_depth",
    "split_required",
    "status",
    "error_message",
]

API_LIMIT = 100
REQUEST_TIMEOUT_SECONDS = 60
MAX_TRANSIENT_RETRIES = 3
MAX_422_TIMEOUT_RETRIES = 2
MAX_429_RETRIES = 3
RETRY_BASE_DELAY_SECONDS = 2.0
REQUEST_DELAY_SECONDS = 0.5
BOUNDARY_OVERLAP_SECONDS = 1
MIN_TIMEOUT_WINDOW = timedelta(days=1)
MIN_CAP_WINDOW = timedelta(seconds=1)
MAX_SPLIT_DEPTH = 30

SUCCESS_STATUS = "complete"
SPLIT_STATUSES = {"split_due_to_cap", "split_due_to_422_timeout"}

KEYWORD_PATTERNS = {
    keyword: re.compile(re.escape(keyword), flags=re.IGNORECASE)
    for keyword in REDDIT_EXTRACTION_KEYWORDS
}


def parse_arguments() -> argparse.Namespace:
    """Parse scope, resume, and validation options."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--year",
        type=int,
        action="append",
        choices=[2021, 2022, 2023],
        help="Restrict extraction to one year; may be repeated.",
    )
    parser.add_argument(
        "--subreddit",
        action="append",
        choices=SUBREDDITS,
        help="Restrict extraction to one subreddit; may be repeated.",
    )
    parser.add_argument(
        "--keyword",
        action="append",
        choices=REDDIT_EXTRACTION_KEYWORDS,
        help="Restrict extraction to one individual keyword; may be repeated.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse checkpointed split decisions and skip successful terminals.",
    )
    return parser.parse_args()


def validate_configuration() -> None:
    """Ensure the extraction universe exactly matches the approved design."""

    expected_subreddits = ["investing", "stocks", "StockMarket"]
    expected_keywords = [
        "Ukraine",
        "Ukrainian",
        "Russia",
        "Russian",
        "Putin",
        "Kyiv",
        "Kiev",
        "Moscow",
        "invasion",
        "sanctions",
        "NATO",
    ]
    if SUBREDDITS != expected_subreddits:
        raise ValueError(
            "SUBREDDITS must contain investing, stocks, and StockMarket only."
        )
    if REDDIT_EXTRACTION_KEYWORDS != expected_keywords:
        raise ValueError(
            "REDDIT_EXTRACTION_KEYWORDS does not match the approved Phase 3A list."
        )
    if START_DATE != "2021-01-01" or END_DATE != "2023-12-31":
        raise ValueError("Phase 3A must cover 2021-01-01 through 2023-12-31.")
    if REQUEST_FIELDS != [
        "id",
        "created_utc",
        "subreddit",
        "title",
        "selftext",
        "score",
        "num_comments",
        "url",
    ]:
        raise ValueError("Only the approved Reddit API fields may be requested.")


def yearly_windows_for_years(years: list[int]) -> list[tuple[datetime, datetime]]:
    """Create one half-open UTC root interval for each selected year."""

    return [
        (
            datetime(year, 1, 1, tzinfo=timezone.utc),
            datetime(year + 1, 1, 1, tzinfo=timezone.utc),
        )
        for year in sorted(set(years))
    ]


def format_utc(timestamp: datetime) -> str:
    """Format a UTC datetime for the API and checkpoint files."""

    return timestamp.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def format_window(window_start: datetime, window_end: datetime) -> str:
    """Format a logical half-open interval."""

    return f"[{format_utc(window_start)}, {format_utc(window_end)})"


def empty_metrics() -> dict[str, int]:
    """Create mutable counters used for progress and tests."""

    return {
        "successful_terminal_windows": 0,
        "api_requests": 0,
        "retries": 0,
        "timeout_422_events": 0,
        "rate_limit_429_events": 0,
        "split_operations": 0,
        "failed_windows": 0,
        "resumed_terminal_windows": 0,
    }


def matched_keywords(title: Any, selftext: Any) -> list[str]:
    """Find every configured extraction term locally in unmodified source text."""

    title_text = "" if title is None or pd.isna(title) else str(title)
    selftext_text = "" if selftext is None or pd.isna(selftext) else str(selftext)
    source_text = f"{title_text}\n{selftext_text}"
    return [
        keyword
        for keyword in REDDIT_EXTRACTION_KEYWORDS
        if KEYWORD_PATTERNS[keyword].search(source_text)
    ]


def enrich_posts(posts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select required source fields and derive local keyword provenance."""

    enriched: list[dict[str, Any]] = []
    for post in posts:
        selected = {field: post.get(field) for field in REQUEST_FIELDS}
        local_matches = matched_keywords(selected["title"], selected["selftext"])
        if not local_matches:
            raise ValueError(
                "Arctic Shift returned post "
                f"{selected['id']!r}, but no configured keyword occurs locally "
                "in title + selftext."
            )
        selected["matched_search_keywords"] = "|".join(local_matches)
        enriched.append(selected)
    return enriched


def query_parameters(
    subreddit: str,
    query_text: str,
    window_start: datetime,
    window_end: datetime,
) -> dict[str, Any]:
    """Build one capped query with a one-second boundary overlap."""

    overlap = timedelta(seconds=BOUNDARY_OVERLAP_SECONDS)
    return {
        "subreddit": subreddit,
        "after": format_utc(window_start - overlap),
        "before": format_utc(window_end + overlap),
        "limit": API_LIMIT,
        "sort": "asc",
        "query": query_text,
        "fields": ",".join(REQUEST_FIELDS),
    }


def parse_api_payload(response: requests.Response) -> list[dict[str, Any]]:
    """Parse either a direct list or Arctic Shift's ``data`` wrapper."""

    payload = response.json()
    if isinstance(payload, dict):
        if "data" not in payload:
            message = payload.get("message") or payload.get("error") or str(payload)
            raise ValueError(f"API response has no data field: {message}")
        payload = payload["data"]
    if not isinstance(payload, list):
        raise ValueError("API response data is not a list.")
    if not all(isinstance(post, dict) for post in payload):
        raise ValueError("API response contains a non-object post record.")
    return payload


def is_arctic_shift_timeout(response: requests.Response) -> bool:
    """Recognize only Arctic Shift's timeout-flavoured HTTP 422 response."""

    return response.status_code == 422 and "timeout" in response.text.lower()


def parse_rate_limit_delay(response: requests.Response, attempt: int) -> float:
    """Return a safe wait based on Arctic Shift's available 429 headers."""

    delays: list[float] = []
    retry_after = response.headers.get("Retry-After")
    if retry_after:
        try:
            delays.append(float(retry_after))
        except ValueError:
            try:
                delays.append(parsedate_to_datetime(retry_after).timestamp() - time.time())
            except (TypeError, ValueError, OverflowError):
                pass
    reset_after = response.headers.get("X-RateLimit-Reset")
    if reset_after:
        try:
            delays.append(float(reset_after))
        except ValueError:
            pass
    reset_at = response.headers.get("X-RateLimit-Reset-At")
    if reset_at:
        try:
            reset_timestamp = float(reset_at)
            if reset_timestamp > 10_000_000_000:
                reset_timestamp /= 1000
            delays.append(reset_timestamp - time.time())
        except ValueError:
            try:
                delays.append(parsedate_to_datetime(reset_at).timestamp() - time.time())
            except (TypeError, ValueError, OverflowError):
                pass
    fallback = RETRY_BASE_DELAY_SECONDS * (attempt + 1)
    positive_delays = [delay for delay in delays if delay > 0]
    return max([fallback, REQUEST_DELAY_SECONDS, *positive_delays]) + 0.25


def request_posts(
    session: requests.Session,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Make one logical API query with status-specific bounded retries."""

    retry_count = 0
    timeout_422_count = 0
    rate_limit_429_count = 0
    transient_failure_count = 0
    api_request_count = 0
    last_status = "failed_unknown"
    last_error = "Request failed without a response."

    while True:
        api_request_count += 1
        try:
            response = session.get(
                ARCTIC_SHIFT_POSTS_URL,
                params=params,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
        except requests.Timeout as error:
            transient_failure_count += 1
            last_status = "failed_connection_timeout"
            last_error = str(error)
            if transient_failure_count > MAX_TRANSIENT_RETRIES:
                break
            retry_count += 1
            time.sleep(RETRY_BASE_DELAY_SECONDS * transient_failure_count)
            continue
        except requests.ConnectionError as error:
            transient_failure_count += 1
            last_status = "failed_connection"
            last_error = str(error)
            if transient_failure_count > MAX_TRANSIENT_RETRIES:
                break
            retry_count += 1
            time.sleep(RETRY_BASE_DELAY_SECONDS * transient_failure_count)
            continue

        if response.status_code == 200:
            try:
                posts = parse_api_payload(response)
            except (requests.JSONDecodeError, ValueError) as error:
                transient_failure_count += 1
                last_status = "failed_response_parsing"
                last_error = str(error)
                if transient_failure_count > MAX_TRANSIENT_RETRIES:
                    break
                retry_count += 1
                time.sleep(RETRY_BASE_DELAY_SECONDS * transient_failure_count)
                continue
            time.sleep(REQUEST_DELAY_SECONDS)
            return {
                "posts": posts,
                "status": "success",
                "error_message": "",
                "api_request_count": api_request_count,
                "retry_count": retry_count,
                "timeout_422_count": timeout_422_count,
                "rate_limit_429_count": rate_limit_429_count,
            }

        if is_arctic_shift_timeout(response):
            timeout_422_count += 1
            last_status = "timeout_422"
            last_error = response.text[:500].replace("\n", " ").strip()
            if timeout_422_count > MAX_422_TIMEOUT_RETRIES:
                break
            retry_count += 1
            delay = RETRY_BASE_DELAY_SECONDS * timeout_422_count
            print(f"Arctic Shift 422 timeout; retrying in {delay:.1f} seconds...")
            time.sleep(delay)
            continue

        if response.status_code == 429:
            rate_limit_429_count += 1
            last_status = "failed_http_429"
            last_error = response.text[:500].replace("\n", " ").strip()
            if rate_limit_429_count > MAX_429_RETRIES:
                break
            retry_count += 1
            delay = parse_rate_limit_delay(response, rate_limit_429_count - 1)
            print(f"Rate limited; waiting {delay:.1f} seconds before retrying...")
            time.sleep(delay)
            continue

        if 500 <= response.status_code <= 599:
            transient_failure_count += 1
            last_status = f"failed_http_{response.status_code}"
            last_error = response.text[:500].replace("\n", " ").strip()
            if transient_failure_count > MAX_TRANSIENT_RETRIES:
                break
            retry_count += 1
            time.sleep(RETRY_BASE_DELAY_SECONDS * transient_failure_count)
            continue

        last_status = f"failed_http_{response.status_code}"
        last_error = response.text[:500].replace("\n", " ").strip()
        break

    time.sleep(REQUEST_DELAY_SECONDS)
    return {
        "posts": None,
        "status": last_status,
        "error_message": last_error,
        "api_request_count": api_request_count,
        "retry_count": retry_count,
        "timeout_422_count": timeout_422_count,
        "rate_limit_429_count": rate_limit_429_count,
    }


def append_csv_row(path: Path, columns: list[str], row: dict[str, Any]) -> None:
    """Append one durable checkpoint row, adding a header when necessary."""

    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
        if needs_header:
            writer.writeheader()
        writer.writerow({column: row.get(column, "") for column in columns})
        output.flush()
        os.fsync(output.fileno())


def append_checkpoint_posts(path: Path, posts: list[dict[str, Any]]) -> None:
    """Append terminal posts before marking their window successful."""

    if not posts:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(
            output, fieldnames=CHECKPOINT_POST_COLUMNS, extrasaction="ignore"
        )
        if needs_header:
            writer.writeheader()
        for post in posts:
            writer.writerow(
                {column: post.get(column, "") for column in CHECKPOINT_POST_COLUMNS}
            )
        output.flush()
        os.fsync(output.fileno())


def checkpoint_key(
    subreddit: str,
    query_text: str,
    window_start: datetime,
    window_end: datetime,
) -> tuple[str, str, str, str]:
    """Return the exact key used to resume one logical interval."""

    return (
        subreddit,
        query_text,
        format_utc(window_start),
        format_utc(window_end),
    )


def load_checkpoint_index(
    path: Path,
) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    """Load the latest record for each checkpointed logical interval."""

    if not path.exists() or path.stat().st_size == 0:
        return {}
    checkpoint = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = set(CHECKPOINT_WINDOW_COLUMNS) - set(checkpoint.columns)
    if missing:
        raise ValueError(f"Checkpoint window file lacks columns: {sorted(missing)}")
    index: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in checkpoint.to_dict("records"):
        key = (
            row["subreddit"],
            row["query_used"],
            row["window_start_utc"],
            row["window_end_utc_exclusive"],
        )
        index[key] = row
    return index


def checkpoint_row(
    subreddit: str,
    query_text: str,
    initial_start: datetime,
    initial_end: datetime,
    window_start: datetime,
    window_end: datetime,
    split_depth: int,
    request_result: dict[str, Any],
    result_count: int | str,
    status: str,
    error_message: str = "",
) -> dict[str, Any]:
    """Build one auditable window-checkpoint record."""

    return {
        "recorded_at_utc": format_utc(datetime.now(timezone.utc)),
        "subreddit": subreddit,
        "initial_window": format_window(initial_start, initial_end),
        "final_window": format_window(window_start, window_end),
        "window_start_utc": format_utc(window_start),
        "window_end_utc_exclusive": format_utc(window_end),
        "query_used": query_text,
        "result_count": result_count,
        "api_request_count": request_result["api_request_count"],
        "retry_count": request_result["retry_count"],
        "timeout_422_count": request_result["timeout_422_count"],
        "rate_limit_429_count": request_result["rate_limit_429_count"],
        "split_depth": split_depth,
        "split_required": status in SPLIT_STATUSES,
        "status": status,
        "error_message": error_message,
    }


def update_metrics(metrics: dict[str, int], request_result: dict[str, Any]) -> None:
    """Accumulate request-level counters."""

    metrics["api_requests"] += int(request_result["api_request_count"])
    metrics["retries"] += int(request_result["retry_count"])
    metrics["timeout_422_events"] += int(request_result["timeout_422_count"])
    metrics["rate_limit_429_events"] += int(
        request_result["rate_limit_429_count"]
    )


def print_progress(
    subreddit: str,
    query_text: str,
    window_start: datetime,
    window_end: datetime,
    status: str,
    metrics: dict[str, int],
    unique_post_ids: set[str],
) -> None:
    """Print enough live state to assess extraction progress."""

    print(
        f"Progress | r/{subreddit} | keyword={query_text} | "
        f"{format_window(window_start, window_end)} | "
        f"status={status} | successful_terminals="
        f"{metrics['successful_terminal_windows']} | "
        f"requests={metrics['api_requests']} | retries={metrics['retries']} | "
        f"timeouts_422={metrics['timeout_422_events']} | "
        f"rate_limits_429={metrics['rate_limit_429_events']} | "
        f"splits={metrics['split_operations']} | "
        f"unique_posts={len(unique_post_ids)}",
        flush=True,
    )


def filter_to_logical_window(
    posts: list[dict[str, Any]],
    window_start: datetime,
    window_end: datetime,
) -> list[dict[str, Any]]:
    """Remove only overlap artifacts outside the logical interval."""

    selected: list[dict[str, Any]] = []
    start_epoch = window_start.timestamp()
    end_epoch = window_end.timestamp()
    for post in posts:
        try:
            created_utc = float(post.get("created_utc"))
        except (TypeError, ValueError):
            selected.append(post)
            continue
        if start_epoch <= created_utc < end_epoch:
            selected.append(post)
    return selected


def split_midpoint(window_start: datetime, window_end: datetime) -> datetime:
    """Return a deterministic midpoint strictly inside the interval."""

    midpoint = window_start + (window_end - window_start) / 2
    if midpoint <= window_start or midpoint >= window_end:
        raise ValueError("No valid midpoint remains for this interval.")
    return midpoint


def fetch_complete_window(
    session: requests.Session,
    subreddit: str,
    query_text: str,
    initial_start: datetime,
    initial_end: datetime,
    window_start: datetime,
    window_end: datetime,
    metrics: dict[str, int],
    unique_post_ids: set[str],
    failures: list[dict[str, Any]],
    checkpoint_posts_file: Path | None = None,
    checkpoint_windows_file: Path | None = None,
    checkpoint_index: dict[tuple[str, str, str, str], dict[str, Any]] | None = None,
    resume: bool = False,
    split_depth: int = 0,
) -> list[dict[str, Any]]:
    """Retrieve one complete interval, splitting caps and persistent 422 timeouts."""

    key = checkpoint_key(subreddit, query_text, window_start, window_end)
    previous = (checkpoint_index or {}).get(key) if resume else None
    if previous and previous["status"] == SUCCESS_STATUS:
        metrics["successful_terminal_windows"] += 1
        metrics["resumed_terminal_windows"] += 1
        print_progress(
            subreddit,
            query_text,
            window_start,
            window_end,
            "resumed_complete",
            metrics,
            unique_post_ids,
        )
        return []

    if previous and previous["status"] in SPLIT_STATUSES:
        metrics["split_operations"] += 1
        midpoint = split_midpoint(window_start, window_end)
        left = fetch_complete_window(
            session, subreddit, query_text, initial_start, initial_end,
            window_start, midpoint, metrics, unique_post_ids, failures,
            checkpoint_posts_file, checkpoint_windows_file, checkpoint_index,
            resume, split_depth + 1,
        )
        right = fetch_complete_window(
            session, subreddit, query_text, initial_start, initial_end,
            midpoint, window_end, metrics, unique_post_ids, failures,
            checkpoint_posts_file, checkpoint_windows_file, checkpoint_index,
            resume, split_depth + 1,
        )
        return left + right

    result = request_posts(
        session,
        query_parameters(subreddit, query_text, window_start, window_end),
    )
    update_metrics(metrics, result)

    if result["status"] == "success":
        posts = result["posts"]
        if len(posts) == API_LIMIT:
            if (
                window_end - window_start <= MIN_CAP_WINDOW
                or split_depth >= MAX_SPLIT_DEPTH
            ):
                row = checkpoint_row(
                    subreddit, query_text, initial_start, initial_end,
                    window_start, window_end, split_depth, result, len(posts),
                    "failed_unsplittable_cap",
                    "The interval remained capped at the minimum split size.",
                )
                failures.append(row)
                metrics["failed_windows"] += 1
                if checkpoint_windows_file:
                    append_csv_row(
                        checkpoint_windows_file, CHECKPOINT_WINDOW_COLUMNS, row
                    )
                print_progress(
                    subreddit, query_text, window_start, window_end, row["status"],
                    metrics, unique_post_ids,
                )
                return []

            row = checkpoint_row(
                subreddit, query_text, initial_start, initial_end,
                window_start, window_end, split_depth, result, len(posts),
                "split_due_to_cap",
            )
            metrics["split_operations"] += 1
            if checkpoint_windows_file:
                append_csv_row(
                    checkpoint_windows_file, CHECKPOINT_WINDOW_COLUMNS, row
                )
                if checkpoint_index is not None:
                    checkpoint_index[key] = row
            print_progress(
                subreddit, query_text, window_start, window_end, row["status"],
                metrics, unique_post_ids,
            )
            midpoint = split_midpoint(window_start, window_end)
            left = fetch_complete_window(
                session, subreddit, query_text, initial_start, initial_end,
                window_start, midpoint, metrics, unique_post_ids, failures,
                checkpoint_posts_file, checkpoint_windows_file, checkpoint_index,
                resume, split_depth + 1,
            )
            right = fetch_complete_window(
                session, subreddit, query_text, initial_start, initial_end,
                midpoint, window_end, metrics, unique_post_ids, failures,
                checkpoint_posts_file, checkpoint_windows_file, checkpoint_index,
                resume, split_depth + 1,
            )
            return left + right

        terminal_posts = filter_to_logical_window(posts, window_start, window_end)
        enriched = enrich_posts(terminal_posts)
        if checkpoint_posts_file:
            append_checkpoint_posts(checkpoint_posts_file, enriched)
        row = checkpoint_row(
            subreddit, query_text, initial_start, initial_end,
            window_start, window_end, split_depth, result, len(posts),
            SUCCESS_STATUS,
        )
        if checkpoint_windows_file:
            append_csv_row(checkpoint_windows_file, CHECKPOINT_WINDOW_COLUMNS, row)
            if checkpoint_index is not None:
                checkpoint_index[key] = row
        metrics["successful_terminal_windows"] += 1
        for post in enriched:
            if post.get("id") is not None:
                unique_post_ids.add(str(post["id"]))
        print_progress(
            subreddit, query_text, window_start, window_end, SUCCESS_STATUS,
            metrics, unique_post_ids,
        )
        return enriched

    if result["status"] == "timeout_422":
        if (
            window_end - window_start > MIN_TIMEOUT_WINDOW
            and split_depth < MAX_SPLIT_DEPTH
        ):
            row = checkpoint_row(
                subreddit, query_text, initial_start, initial_end,
                window_start, window_end, split_depth, result, "",
                "split_due_to_422_timeout", result["error_message"],
            )
            metrics["split_operations"] += 1
            if checkpoint_windows_file:
                append_csv_row(
                    checkpoint_windows_file, CHECKPOINT_WINDOW_COLUMNS, row
                )
                if checkpoint_index is not None:
                    checkpoint_index[key] = row
            print_progress(
                subreddit, query_text, window_start, window_end, row["status"],
                metrics, unique_post_ids,
            )
            midpoint = split_midpoint(window_start, window_end)
            left = fetch_complete_window(
                session, subreddit, query_text, initial_start, initial_end,
                window_start, midpoint, metrics, unique_post_ids, failures,
                checkpoint_posts_file, checkpoint_windows_file, checkpoint_index,
                resume, split_depth + 1,
            )
            right = fetch_complete_window(
                session, subreddit, query_text, initial_start, initial_end,
                midpoint, window_end, metrics, unique_post_ids, failures,
                checkpoint_posts_file, checkpoint_windows_file, checkpoint_index,
                resume, split_depth + 1,
            )
            return left + right

        status = "failed_terminal_422_timeout"
        error_message = (
            "Arctic Shift timeout persisted after retries at an interval no wider "
            f"than {MIN_TIMEOUT_WINDOW}. {result['error_message']}"
        )
    else:
        status = result["status"]
        error_message = result["error_message"]

    row = checkpoint_row(
        subreddit, query_text, initial_start, initial_end,
        window_start, window_end, split_depth, result, "", status, error_message,
    )
    failures.append(row)
    metrics["failed_windows"] += 1
    if checkpoint_windows_file:
        append_csv_row(checkpoint_windows_file, CHECKPOINT_WINDOW_COLUMNS, row)
        if checkpoint_index is not None:
            checkpoint_index[key] = row
    print_progress(
        subreddit, query_text, window_start, window_end, status,
        metrics, unique_post_ids,
    )
    return []


def new_session() -> requests.Session:
    """Create the conservative shared HTTP session used by live extraction."""

    session = requests.Session()
    session.headers.update(
        {"User-Agent": "MA-thesis-research-reddit-extraction/2.0"}
    )
    return session


def existing_checkpoint_ids(path: Path) -> set[str]:
    """Load post IDs only, for inexpensive resume progress reporting."""

    if not path.exists() or path.stat().st_size == 0:
        return set()
    return set(
        pd.read_csv(path, usecols=["id"], dtype=str)["id"].dropna()
    )


def ensure_checkpoint_mode(resume: bool) -> None:
    """Prevent accidental mixing when checkpoint files already exist."""

    existing = [
        path
        for path in [REDDIT_CHECKPOINT_POSTS_FILE, REDDIT_CHECKPOINT_WINDOWS_FILE]
        if path.exists() and path.stat().st_size > 0
    ]
    if existing and not resume:
        paths = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            "Checkpoint files already exist. Use --resume to reuse them or move "
            f"them before starting a new extraction: {paths}"
        )


def run_scoped_extraction(
    years: list[int],
    subreddits: list[str],
    keywords: list[str],
    resume: bool,
) -> None:
    """Run individual-keyword extraction for a selected scope."""

    ensure_checkpoint_mode(resume)
    windows = yearly_windows_for_years(years)
    planned_initial_queries = len(subreddits) * len(keywords) * len(windows)
    checkpoint_index = load_checkpoint_index(REDDIT_CHECKPOINT_WINDOWS_FILE)
    unique_post_ids = existing_checkpoint_ids(REDDIT_CHECKPOINT_POSTS_FILE)
    metrics = empty_metrics()
    failures: list[dict[str, Any]] = []
    print(
        f"Starting Phase 3A scope: {len(subreddits)} subreddit(s) x "
        f"{len(keywords)} individual keyword(s) x {len(windows)} yearly "
        f"window(s) = {planned_initial_queries} initial requests before "
        "adaptive splits.",
        flush=True,
    )
    with new_session() as session:
        for subreddit in subreddits:
            for keyword in keywords:
                for window_start, window_end in windows:
                    fetch_complete_window(
                        session, subreddit, keyword,
                        window_start, window_end, window_start, window_end,
                        metrics, unique_post_ids, failures,
                        REDDIT_CHECKPOINT_POSTS_FILE,
                        REDDIT_CHECKPOINT_WINDOWS_FILE,
                        checkpoint_index, resume,
                    )

    print("\nScoped extraction checkpoint report")
    for name, value in metrics.items():
        print(f"{name}: {value}")
    print(f"unique checkpoint post IDs: {len(unique_post_ids)}")
    if failures:
        raise RuntimeError(
            f"The selected scope has {len(failures)} failed terminal intervals."
        )
    full_scope = (
        set(years) == {2021, 2022, 2023}
        and set(subreddits) == set(SUBREDDITS)
        and set(keywords) == set(REDDIT_EXTRACTION_KEYWORDS)
    )
    if full_scope:
        finalize_full_extraction()
    else:
        print(
            "Scoped run completed and checkpointed. Final raw outputs are deferred "
            "until a complete all-year, all-subreddit resume run validates coverage."
        )


def first_non_missing(values: pd.Series) -> Any:
    """Keep the first source value, using a later retrieval only if missing."""

    for value in values:
        if value is not None and not pd.isna(value):
            return value
    return pd.NA


def deduplicate_posts(
    posts: list[dict[str, Any]], extracted_at_utc: str
) -> pd.DataFrame:
    """Deduplicate globally by post ID and union locally matched keywords."""

    if not posts:
        raise ValueError("No Reddit posts were supplied for final deduplication.")
    raw = pd.DataFrame(posts)
    missing = set(CHECKPOINT_POST_COLUMNS) - set(raw.columns)
    if missing:
        raise ValueError(f"Reddit rows lack columns: {sorted(missing)}")
    if raw["id"].isna().any() or raw["id"].astype(str).str.strip().eq("").any():
        raise ValueError("At least one Reddit row lacks a post ID.")
    keyword_order = {
        keyword: position
        for position, keyword in enumerate(REDDIT_EXTRACTION_KEYWORDS)
    }
    rows: list[dict[str, Any]] = []
    for post_id, group in raw.groupby("id", sort=False, dropna=False):
        subreddits = group["subreddit"].dropna().astype(str).unique()
        if len(subreddits) > 1:
            raise ValueError(
                f"Reddit ID {post_id} has conflicting subreddits: "
                f"{subreddits.tolist()}"
            )
        row = {field: first_non_missing(group[field]) for field in REQUEST_FIELDS}
        keyword_values: set[str] = set()
        for value in group["matched_search_keywords"].dropna().astype(str):
            keyword_values.update(part for part in value.split("|") if part)
        ordered = sorted(
            keyword_values,
            key=lambda value: (keyword_order.get(value, len(keyword_order)), value),
        )
        row["matched_search_keywords"] = "|".join(ordered)
        rows.append(row)
    data = pd.DataFrame(rows)
    data["created_utc"] = pd.to_numeric(data["created_utc"], errors="coerce")
    data["created_datetime_utc"] = pd.to_datetime(
        data["created_utc"], unit="s", errors="coerce", utc=True
    )
    data["date_utc"] = data["created_datetime_utc"].dt.date
    data["extraction_source"] = ARCTIC_SHIFT_SOURCE_NAME
    data["source_endpoint"] = ARCTIC_SHIFT_POSTS_URL
    data["extracted_at_utc"] = extracted_at_utc
    sample_start = pd.Timestamp(START_DATE, tz="UTC")
    sample_end = pd.Timestamp(END_DATE, tz="UTC") + pd.Timedelta(days=1)
    data = data.loc[
        data["created_datetime_utc"].ge(sample_start)
        & data["created_datetime_utc"].lt(sample_end)
    ].copy()
    return (
        data.loc[:, RAW_OUTPUT_COLUMNS]
        .sort_values(["created_datetime_utc", "id"], kind="stable")
        .reset_index(drop=True)
    )


def text_counts(data: pd.DataFrame) -> dict[str, int]:
    """Count source-text states without removing or modifying them."""

    title = data["title"].astype("string")
    selftext = data["selftext"].astype("string")
    stripped = selftext.str.strip()
    return {
        "missing_title_count": int(title.isna().sum()),
        "blank_selftext_count": int((selftext.isna() | stripped.eq("")).sum()),
        "removed_selftext_count": int(stripped.eq("[removed]").sum()),
        "deleted_selftext_count": int(stripped.eq("[deleted]").sum()),
    }


def validate_raw_dataset(data: pd.DataFrame) -> None:
    """Apply Phase 3A final-corpus validation checks."""

    if data.empty or list(data.columns) != RAW_OUTPUT_COLUMNS:
        raise ValueError("The raw Reddit dataset is empty or has an unexpected schema.")
    if set(data["subreddit"].dropna().unique()) - set(SUBREDDITS):
        raise ValueError("An unapproved subreddit occurs in the raw dataset.")
    if data["subreddit"].isna().any() or data["id"].duplicated().any():
        raise ValueError("Subreddits must exist and Reddit IDs must be unique.")
    if data["created_utc"].isna().any() or data["created_datetime_utc"].isna().any():
        raise ValueError("At least one Reddit timestamp is invalid.")
    if not isinstance(data["created_datetime_utc"].dtype, pd.DatetimeTZDtype):
        raise ValueError("created_datetime_utc is not timezone-aware.")
    if str(data["created_datetime_utc"].dt.tz) != "UTC":
        raise ValueError("created_datetime_utc is not UTC.")
    sample_start = pd.Timestamp(START_DATE, tz="UTC")
    sample_end = pd.Timestamp(END_DATE, tz="UTC") + pd.Timedelta(days=1)
    if not (
        data["created_datetime_utc"].ge(sample_start)
        & data["created_datetime_utc"].lt(sample_end)
    ).all():
        raise ValueError("At least one post lies outside the thesis period.")
    if not data["created_datetime_utc"].is_monotonic_increasing:
        raise ValueError("Raw Reddit observations are not sorted ascending.")
    if data["matched_search_keywords"].isna().any() or data[
        "matched_search_keywords"
    ].astype(str).str.strip().eq("").any():
        raise ValueError("At least one post lacks local keyword provenance.")
    if not data["source_endpoint"].eq(ARCTIC_SHIFT_POSTS_URL).all():
        raise ValueError("A non-Arctic-Shift source was mixed into the dataset.")


def build_extraction_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Create a complete subreddit-by-year sample-composition table."""

    composition = data.assign(
        year=data["created_datetime_utc"].dt.year.astype(int)
    ).groupby(["year", "subreddit"])["id"].nunique()
    complete_index = pd.MultiIndex.from_product(
        [range(2021, 2024), SUBREDDITS], names=["year", "subreddit"]
    )
    return (
        composition.reindex(complete_index, fill_value=0)
        .rename("unique_posts")
        .reset_index()
    )


def validate_complete_checkpoint_coverage(checkpoint: pd.DataFrame) -> None:
    """Confirm every planned yearly keyword root resolves to successful terminals."""

    latest_index = load_checkpoint_index(REDDIT_CHECKPOINT_WINDOWS_FILE)

    def interval_complete(
        subreddit: str,
        keyword: str,
        start: datetime,
        end: datetime,
    ) -> bool:
        row = latest_index.get(
            checkpoint_key(subreddit, keyword, start, end)
        )
        if not row:
            return False
        if row["status"] == SUCCESS_STATUS:
            return True
        if row["status"] not in SPLIT_STATUSES:
            return False
        midpoint = split_midpoint(start, end)
        return interval_complete(
            subreddit, keyword, start, midpoint
        ) and interval_complete(subreddit, keyword, midpoint, end)

    incomplete = []
    for subreddit in SUBREDDITS:
        for keyword in REDDIT_EXTRACTION_KEYWORDS:
            for start, end in yearly_windows_for_years([2021, 2022, 2023]):
                if not interval_complete(subreddit, keyword, start, end):
                    incomplete.append(
                        (subreddit, keyword, format_window(start, end))
                    )
    if incomplete:
        examples = "; ".join(
            f"r/{sub} {keyword} {window}"
            for sub, keyword, window in incomplete[:5]
        )
        raise ValueError(
            f"Checkpoint coverage is incomplete for {len(incomplete)} initial "
            f"windows. Examples: {examples}"
        )
    if checkpoint.empty:
        raise ValueError("The window checkpoint is empty.")


def finalize_full_extraction() -> None:
    """Build final Phase 3A outputs only from complete checkpoint coverage."""

    windows = pd.read_csv(REDDIT_CHECKPOINT_WINDOWS_FILE, keep_default_na=False)
    validate_complete_checkpoint_coverage(windows)
    posts = pd.read_csv(
        REDDIT_CHECKPOINT_POSTS_FILE,
        dtype={"id": str, "subreddit": str},
        keep_default_na=False,
    ).to_dict("records")
    data = deduplicate_posts(posts, format_utc(datetime.now(timezone.utc)))
    validate_raw_dataset(data)
    summary = build_extraction_summary(data)
    RAW_REDDIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REDDIT_EXTRACTION_SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)
    REDDIT_QUERY_SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(RAW_REDDIT_FILE, index=False, encoding="utf-8-sig")
    summary.to_csv(REDDIT_EXTRACTION_SUMMARY_FILE, index=False)
    windows.to_csv(REDDIT_QUERY_SUMMARY_FILE, index=False)
    print(f"Saved final raw candidate data to {RAW_REDDIT_FILE}")
    print(f"Saved extraction summary to {REDDIT_EXTRACTION_SUMMARY_FILE}")
    print(f"Saved query diagnostics to {REDDIT_QUERY_SUMMARY_FILE}")


def main() -> None:
    """Run a checkpointed, optionally scoped individual-keyword extraction."""

    validate_configuration()
    arguments = parse_arguments()
    years = arguments.year or [2021, 2022, 2023]
    subreddits = arguments.subreddit or SUBREDDITS
    keywords = arguments.keyword or REDDIT_EXTRACTION_KEYWORDS
    run_scoped_extraction(years, subreddits, keywords, arguments.resume)


if __name__ == "__main__":
    main()
