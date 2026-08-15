#test_sentimentindex
#procedure:raw Reddit titles → FinBERT labels → numeric sentiment scores → daily sentiment index

import pandas as pd
from transformers import pipeline

# --------------------------------------------------
# 1. File names
# --------------------------------------------------

reddit_file = "reddit_large_test_raw.csv"
output_file = "daily_reddit_sentiment_index_test.csv"

# --------------------------------------------------
# 2. Load Reddit data
# --------------------------------------------------

df = pd.read_csv(reddit_file)

print("Loaded Reddit data:")
print(df.head())
print("\nColumns:")
print(df.columns)

# --------------------------------------------------
# 3. Keep only the columns needed for the first test
# --------------------------------------------------

df = df[["date", "title"]].copy()

# Remove rows with missing titles or dates
df = df.dropna(subset=["date", "title"])

# Convert title to string
df["title"] = df["title"].astype(str)

# Remove empty or very short titles
df = df[df["title"].str.strip().str.len() > 5]

# Convert date.
# Your screenshot shows day/month/year format, for example 03/01/2021.
df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce").dt.date

# Remove rows where date conversion failed
df = df.dropna(subset=["date"])

print("\nCleaned Reddit data:")
print(df.head())
print("\nNumber of Reddit titles to score:", len(df))

# --------------------------------------------------
# 4. Optional test limit
# --------------------------------------------------
# Keep this while testing so the script runs fast.
# Later, comment this line out to run the full file.

df = df.head(100)

print("\nRunning FinBERT on this many titles:", len(df))

# --------------------------------------------------
# 5. Load FinBERT
# --------------------------------------------------

sentiment_model = pipeline(
    "sentiment-analysis",
    model="ProsusAI/finbert"
)

# --------------------------------------------------
# 6. Score titles
# --------------------------------------------------

def score_title(title):
    result = sentiment_model(title[:512])[0]

    label = result["label"].lower()
    confidence = result["score"]

    if label == "positive":
        numeric_score = confidence
    elif label == "negative":
        numeric_score = -confidence
    else:
        numeric_score = 0

    return label, confidence, numeric_score


labels = []
confidences = []
numeric_scores = []

for i, title in enumerate(df["title"], start=1):
    label, confidence, numeric_score = score_title(title)

    labels.append(label)
    confidences.append(confidence)
    numeric_scores.append(numeric_score)

    if i % 10 == 0:
        print(f"Scored {i} titles...")

df["finbert_label"] = labels
df["finbert_confidence"] = confidences
df["sentiment_score"] = numeric_scores

print("\nExample scored titles:")
print(df[["date", "title", "finbert_label", "finbert_confidence", "sentiment_score"]].head(10))

# --------------------------------------------------
# 7. Build daily sentiment index
# --------------------------------------------------

daily_sentiment = df.groupby("date").agg(
    avg_sentiment_score=("sentiment_score", "mean"),
    median_sentiment_score=("sentiment_score", "median"),
    post_count=("title", "count"),
    positive_count=("finbert_label", lambda x: (x == "positive").sum()),
    negative_count=("finbert_label", lambda x: (x == "negative").sum()),
    neutral_count=("finbert_label", lambda x: (x == "neutral").sum())
).reset_index()

# Optional extra index:
# Share of positive posts minus share of negative posts
daily_sentiment["pos_minus_neg_share"] = (
    daily_sentiment["positive_count"] / daily_sentiment["post_count"]
    - daily_sentiment["negative_count"] / daily_sentiment["post_count"]
)

# --------------------------------------------------
# 8. Save result
# --------------------------------------------------

daily_sentiment.to_csv(output_file, index=False)

print("\nDaily sentiment index:")
print(daily_sentiment.head())

print(f"\nSaved file: {output_file}")