#reddit_largeextraction 
import requests
import pandas as pd
import time

BASE_URL = "https://arctic-shift.photon-reddit.com/api/posts/search"

# Start with a focused list. We can expand later. #remove hashtages from the list names to activate them
SUBREDDITS = [
    "investing",
    "stocks",
    "StockMarket",
    #"economics",
    #"Europe",
    #"worldnews"
]

KEYWORDS = [
    "Ukraine",
    "Russia",
    #"war",
    #"sanctions",
    #"energy crisis",
    #"gas prices",
    #"inflation",
    #"recession",
    #"oil prices"
]

FIELDS = "id,created_utc,subreddit,title,score,num_comments,url,selftext" #can later add selftext (body), author, etc. if needed. For now, we want to keep it simple and focused on titles and metadata.

# Monthly windows from Jan 2021 to Dec 2023, changable for future use. This is a good way to avoid rate limits and large data pulls.
#Change the start and end dates as needed. The frequency is set to "MS" for month start.
MONTHS = pd.date_range(start="2021-01-01", end="2022-12-31", freq="MS")

all_posts = []

for subreddit in SUBREDDITS:
    for keyword in KEYWORDS:
        for month_start in MONTHS:
            month_end = month_start + pd.offsets.MonthEnd(1)

            after_date = month_start.strftime("%Y-%m-%d")
            before_date = month_end.strftime("%Y-%m-%d")

            print(f"Searching r/{subreddit} | {keyword} | {after_date} to {before_date}")

            params = {
                "subreddit": subreddit,
                "after": after_date,
                "before": before_date,
                "limit": 20, #can be adjusted as needed, but 100 is a good starting point
                "sort": "asc",
                "query": keyword,
                "fields": FIELDS
            }

            try:
                response = requests.get(BASE_URL, params=params, timeout=60)

                if response.status_code != 200:
                    print(f"Skipped because status code was {response.status_code}")
                    continue

                data = response.json()

                if isinstance(data, dict):
                    # Sometimes APIs return a dict with data inside it.
                    # This handles that safely.
                    data = data.get("data", [])

                for post in data:
                    post["search_keyword"] = keyword
                    post["search_month"] = after_date
                    all_posts.append(post)

                # Be polite to the server
                time.sleep(1)

            except Exception as e:
                print(f"Error: {e}")
                continue

df = pd.DataFrame(all_posts)

if not df.empty:
    df = df.drop_duplicates(subset=["id"])

    # Convert Reddit timestamp into readable date
    df["created_utc"] = pd.to_numeric(df["created_utc"], errors="coerce")
    df["created_datetime"] = pd.to_datetime(df["created_utc"], unit="s", errors="coerce")
    df["date"] = df["created_datetime"].dt.date

    # Basic cleaning
    df["title"] = df["title"].fillna("")  
    df["selftext"] = df["selftext"].fillna("") #removed for now, but can be added back if we want to analyze post bodies later.

    df = df.sort_values("created_datetime")

df.to_csv("reddit_large_test_raw.csv", index=False, encoding="utf-8-sig")

print("Finished.")
print("Number of unique posts:", len(df))
print("Saved file: reddit_large_test_raw.csv")