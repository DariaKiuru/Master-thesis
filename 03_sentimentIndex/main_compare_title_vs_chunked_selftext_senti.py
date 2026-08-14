#compare_title_vs_chunked_selftext_sentiment

import re
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# --------------------------------------------------
# 1. File names
# --------------------------------------------------

reddit_file = "reddit_large_test_raw.csv"
output_file = "reddit_title_vs_chunked_selftext_sentiment_test.csv"

# --------------------------------------------------
# 2. Load Reddit data
# --------------------------------------------------

df = pd.read_csv(reddit_file)

print("Available columns:")
print(df.columns)

df = df[["date", "title", "selftext"]].copy()

df = df.dropna(subset=["date", "title"])
df["title"] = df["title"].astype(str)
df["selftext"] = df["selftext"].fillna("").astype(str)

df["selftext"] = df["selftext"].replace(
    ["[removed]", "[deleted]", "removed", "deleted"],
    ""
)

df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce").dt.date
df = df.dropna(subset=["date"])

df = df[df["title"].str.strip().str.len() > 5]

# Test only first 100 rows for now
df = df.head(100)

print("Rows used for test:", len(df))

# --------------------------------------------------
# 3. Load FinBERT
# --------------------------------------------------

model_name = "ProsusAI/finbert"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

sentiment_model = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    top_k=None,
    truncation=True,
    max_length=512
)

# --------------------------------------------------
# 4. Helper functions
# --------------------------------------------------

def split_into_chunks(text, max_words=30, max_chunks=120):
    """
    Split long Reddit selftext into smaller chunks.
    max_words keeps chunks short enough for FinBERT.
    max_chunks prevents one huge post from taking too long.
    """
    text = str(text)
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)

    chunks = []
    current_chunk = []
    current_words = 0

    for sentence in sentences:
        word_count = len(sentence.split())

        if current_words + word_count > max_words and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_words = word_count
        else:
            current_chunk.append(sentence)
            current_words += word_count

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks[:max_chunks]


def score_short_text(text):
    """
    Score one short text using all FinBERT probabilities.
    Sentiment score = P(positive) - P(negative)
    """
    result = sentiment_model(str(text))[0]

    probs = {item["label"].lower(): item["score"] for item in result}

    positive_prob = probs.get("positive", 0)
    negative_prob = probs.get("negative", 0)
    neutral_prob = probs.get("neutral", 0)

    sentiment_score = positive_prob - negative_prob

    top_label = max(probs, key=probs.get)
    top_confidence = probs[top_label]

    return {
        "top_label": top_label,
        "top_confidence": top_confidence,
        "positive_prob": positive_prob,
        "negative_prob": negative_prob,
        "neutral_prob": neutral_prob,
        "sentiment_score": sentiment_score
    }


def score_long_text(text):
    """
    Score a long Reddit post by splitting it into chunks.
    The final score is the average of chunk sentiment scores.
    """
    chunks = split_into_chunks(text)

    if len(chunks) == 0:
        return {
            "chunked_label": "empty",
            "chunked_sentiment_score": np.nan,
            "chunk_count": 0
        }

    chunk_scores = []
    labels = []

    for chunk in chunks:
        result = score_short_text(chunk)
        chunk_scores.append(result["sentiment_score"])
        labels.append(result["top_label"])

    avg_score = np.mean(chunk_scores)

    if avg_score > 0.05:
        final_label = "positive"
    elif avg_score < -0.05:
        final_label = "negative"
    else:
        final_label = "neutral"

    return {
        "chunked_label": final_label,
        "chunked_sentiment_score": avg_score,
        "chunk_count": len(chunks)
    }

# --------------------------------------------------
# 5. Score title only
# --------------------------------------------------

title_results = []

for i, title in enumerate(df["title"], start=1):
    result = score_short_text(title)
    title_results.append(result)

    if i % 10 == 0:
        print(f"Scored {i} titles...")

df["title_label"] = [r["top_label"] for r in title_results]
df["title_sentiment_score"] = [r["sentiment_score"] for r in title_results]
df["title_positive_prob"] = [r["positive_prob"] for r in title_results]
df["title_negative_prob"] = [r["negative_prob"] for r in title_results]
df["title_neutral_prob"] = [r["neutral_prob"] for r in title_results]

# --------------------------------------------------
# 6. Score title + selftext as chunked text
# --------------------------------------------------

df["combined_text"] = df["title"].str.strip() + ". " + df["selftext"].str.strip()

combined_results = []

for i, text in enumerate(df["combined_text"], start=1):
    result = score_long_text(text)
    combined_results.append(result)

    if i % 10 == 0:
        print(f"Scored {i} title+selftext posts...")

df["chunked_label"] = [r["chunked_label"] for r in combined_results]
df["chunked_sentiment_score"] = [r["chunked_sentiment_score"] for r in combined_results]
df["chunk_count"] = [r["chunk_count"] for r in combined_results]

# --------------------------------------------------
# 7. Compare
# --------------------------------------------------

df["label_changed"] = df["title_label"] != df["chunked_label"]
df["sentiment_difference"] = df["chunked_sentiment_score"] - df["title_sentiment_score"]

print("\nLabel counts: title only")
print(df["title_label"].value_counts())

print("\nLabel counts: title + chunked selftext")
print(df["chunked_label"].value_counts())

print("\nHow often did the label change?")
print(df["label_changed"].value_counts())

print("\nAverage sentiment: title only")
print(df["title_sentiment_score"].mean())

print("\nAverage sentiment: title + chunked selftext")
print(df["chunked_sentiment_score"].mean())

print("\nExamples where label changed:")
print(
    df[df["label_changed"] == True][
        [
            "date",
            "title",
            "title_label",
            "chunked_label",
            "title_sentiment_score",
            "chunked_sentiment_score",
            "sentiment_difference",
            "chunk_count"
        ]
    ].head(20)
)

# --------------------------------------------------
# 8. Save
# --------------------------------------------------

df.to_csv(output_file, index=False)

print(f"\nSaved file: {output_file}")