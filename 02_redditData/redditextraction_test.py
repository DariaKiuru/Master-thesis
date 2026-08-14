#redditextraction_test
import requests
import pandas as pd
import time

BASE_URL = "https://arctic-shift.photon-reddit.com/api/posts/search"

subreddit = "investing"
keyword = "Ukraine"

params = {
    "subreddit": subreddit,
    "after": "2022-02-24",
    "before": "2022-03-31",
    "limit": 25,
    "sort": "asc",
    "query": keyword,
    "fields": "id,created_utc,subreddit,title,score,num_comments,url"
}

response = requests.get(BASE_URL, params=params, timeout=30)

print("Status code:", response.status_code)

data = response.json()

df = pd.DataFrame(data)

print(df.head())

df.to_csv("reddit_titles_test.csv", index=False, encoding="utf-8-sig")

print("Saved file: reddit_titles_test.csv")