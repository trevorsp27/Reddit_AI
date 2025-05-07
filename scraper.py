import praw
import pandas as pd
import re
from datetime import datetime
from tqdm import tqdm
import time

#reddit API
reddit = praw.Reddit()

subreddits_to_scrape = [
    "pennystocks", "stocks", "wallstreetbets",
    "investing", "StockMarket", "options", "RobinHoodPennyStocks"
]
posts_per_subreddit = 1000
output_file = "reddit_posts_large.csv"

def extract_tickers(text):
    return re.findall(r'\b[A-Z]{2,5}\b', text)

#scraper
def scrape_reddit_posts(subreddit_name, limit):
    posts = []
    print(f"\n🔍 Scraping r/{subreddit_name}...")

    try:
        subreddit = reddit.subreddit(subreddit_name)
        for post in tqdm(subreddit.hot(limit=limit), desc=f"r/{subreddit_name}"):
            if post.selftext in ["[removed]", "[deleted]"]:
                continue

            combined_text = (post.title or "") + " " + (post.selftext or "")
            tickers = extract_tickers(combined_text)

            if not tickers:
                continue

            posts.append({
                "id": post.id,
                "subreddit": subreddit_name,
                "title": post.title,
                "body": post.selftext,
                "date": datetime.fromtimestamp(post.created_utc),
                "upvotes": post.score,
                "comments": post.num_comments,
                "tickers": list(set(tickers))
            })

            time.sleep(0.1)  #reddit API needs time to load

    except Exception as e:
        print(f"failed to scrape r/{subreddit_name}: {e}")

    return posts

#run
all_posts = []

for sub in subreddits_to_scrape:
    posts = scrape_reddit_posts(sub, limit=posts_per_subreddit)
    all_posts.extend(posts)

#save to file
df = pd.DataFrame(all_posts).drop_duplicates(subset=["id"])
df.to_csv(output_file, index=False)
print(f"saved {len(df)} posts to {output_file}")
