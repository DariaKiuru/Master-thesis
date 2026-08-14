#compare_title_vs_selftext_sentiment
import pandas as pd
from transformers import pipeline

# --------------------------------------------------
# 1. File names
# --------------------------------------------------

reddit_file = "reddit_large_test_raw.csv"
output_file = "reddit_title_vs_selftext_sentiment_test.csv"

# --------------------------------------------------
# 2. Load data
# --------------------------------------------------

df = pd.read_csv(reddit_file)

print("Available columns:")
print(df.columns)

# --------------------------------------------------
# 3. Keep needed columns
# --------------------------------------------------

df = df[["date", "title", "selftext"]].copy()

# --------------------------------------------------
# 4. Clean data
# --------------------------------------------------

df = df.dropna(subset=["date", "title"])

df["title"] = df["title"].astype(str)
df["selftext"] = df["selftext"].fillna("").astype(str)

# Remove common Reddit placeholder text
df["selftext"] = df["selftext"].replace(
    ["[removed]", "[deleted]", "removed", "deleted"],
    ""
)

# Convert date
df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce").dt.date
df = df.dropna(subset=["date"])

# Remove very short titles
df = df[df["title"].str.strip().str.len() > 5]

# --------------------------------------------------
# 5. Test sample
# --------------------------------------------------
# Keep this small first so you do not wait too long.
# Later increase to 500, 1000, or comment it out.

df = df.head(100)

print("\nRows used for test:", len(df))

# --------------------------------------------------
# 6. Create two text versions
# --------------------------------------------------

df["text_title_only"] = df["title"].str.strip()

df["text_title_plus_selftext"] = (
    df["title"].str.strip()
    + ". "
    + df["selftext"].str.strip()
)

# --------------------------------------------------
# 7. Load FinBERT
# --------------------------------------------------

sentiment_model = pipeline(
    "sentiment-analysis",
    model="ProsusAI/finbert"
)

# --------------------------------------------------
# 8. Sentiment scoring function
# --------------------------------------------------

def score_text(text):
    text = str(text)

    # BERT/FinBERT has a length limit.
    # For this first test, we cut long text.
    text = text[:512]

    result = sentiment_model(text)[0]

    label = result["label"].lower()
    confidence = result["score"]

    if label == "positive":
        numeric_score = confidence
    elif label == "negative":
        numeric_score = -confidence
    else:
        numeric_score = 0

    return label, confidence, numeric_score

# --------------------------------------------------
# 9. Score title only
# --------------------------------------------------

title_labels = []
title_confidences = []
title_scores = []

for i, text in enumerate(df["text_title_only"], start=1):
    label, confidence, score = score_text(text)

    title_labels.append(label)
    title_confidences.append(confidence)
    title_scores.append(score)

    if i % 10 == 0:
        print(f"Scored {i} titles...")

df["title_label"] = title_labels
df["title_confidence"] = title_confidences
df["title_sentiment_score"] = title_scores

# --------------------------------------------------
# 10. Score title + selftext
# --------------------------------------------------

combined_labels = []
combined_confidences = []
combined_scores = []

for i, text in enumerate(df["text_title_plus_selftext"], start=1):
    label, confidence, score = score_text(text)

    combined_labels.append(label)
    combined_confidences.append(confidence)
    combined_scores.append(score)

    if i % 10 == 0:
        print(f"Scored {i} title+selftext texts...")

df["title_selftext_label"] = combined_labels
df["title_selftext_confidence"] = combined_confidences
df["title_selftext_sentiment_score"] = combined_scores

# --------------------------------------------------
# 11. Compare title vs title+selftext
# --------------------------------------------------

df["label_changed"] = df["title_label"] != df["title_selftext_label"]

df["sentiment_difference"] = (
    df["title_selftext_sentiment_score"] - df["title_sentiment_score"]
)

print("\nLabel counts: title only")
print(df["title_label"].value_counts())

print("\nLabel counts: title + selftext")
print(df["title_selftext_label"].value_counts())

print("\nHow often did the label change?")
print(df["label_changed"].value_counts())

print("\nAverage sentiment: title only")
print(df["title_sentiment_score"].mean())

print("\nAverage sentiment: title + selftext")
print(df["title_selftext_sentiment_score"].mean())

print("\nExamples where label changed:")
changed = df[df["label_changed"] == True]

print(
    changed[
        [
            "date",
            "title",
            "title_label",
            "title_selftext_label",
            "title_sentiment_score",
            "title_selftext_sentiment_score",
            "sentiment_difference"
        ]
    ].head(20)
)

# --------------------------------------------------
# 12. Save detailed comparison
# --------------------------------------------------

df.to_csv(output_file, index=False)

print(f"\nSaved detailed comparison file: {output_file}")
