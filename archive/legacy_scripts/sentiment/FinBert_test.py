#FinBert_test
from transformers import pipeline

# Load FinBERT model
finbert = pipeline("text-classification", model="ProsusAI/finbert")

# Example financial texts
texts = [
    "2021 will be an unforgettable year for uranium investors (15 powerful catalyst that are in place",
    "Markets recovered after signs of diplomatic progress.",
    "Investors are waiting for the next ECB decision."
]

# Run FinBERT
results = finbert(texts)

# Show results
for text, result in zip(texts, results):
    print("Text:", text)
    print("Sentiment:", result["label"])
    print("Confidence:", round(result["score"], 4))
    print()
