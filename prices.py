import pandas as pd
import yfinance as yf
from datetime import timedelta
import ast
import re
import json

# === Load real tickers ===
with open("company_tickers.json") as f:
    ticker_data = json.load(f)
    VALID_TICKERS = {v["ticker"].upper() for v in ticker_data.values()}

def clean_tickers(ticker_list):
    """Filter valid US-style stock tickers only."""
    clean = []
    for t in ticker_list:
        t = t.strip().upper()
        if re.fullmatch(r'[A-Z]{1,5}(-[A-Z])?', t) and t in VALID_TICKERS:
            clean.append(t)
    return clean

def get_price_change(ticker, post_date):
    try:
        start = post_date - timedelta(days=7)
        end = post_date + timedelta(days=7)
        data = yf.download(ticker, start=start.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'), progress=False)

        if data.empty or len(data['Close']) < 2:
            return None

        price_before = data['Close'].iloc[0]
        price_after = data['Close'].iloc[-1]
        percent_change = ((price_after - price_before) / price_before) * 100
        return round(percent_change, 2)
    except Exception as e:
        print(f"[⚠️] Error fetching {ticker}: {e}")
        return None

def enrich_data(input_csv, output_csv):
    df = pd.read_csv(input_csv, parse_dates=['date'])

    df['main_ticker'] = None
    df['price_change'] = None
    df['reliable'] = None

    skipped = []

    for i, row in df.iterrows():
        try:
            raw_tickers = ast.literal_eval(row['tickers']) if isinstance(row['tickers'], str) else []
            clean = clean_tickers(raw_tickers)
            if not clean:
                skipped.append((row['id'], raw_tickers))
                continue

            main_ticker = clean[0]
            change = get_price_change(main_ticker, row['date'])

            if change is not None:
                df.at[i, 'main_ticker'] = main_ticker
                df.at[i, 'price_change'] = change
                df.at[i, 'reliable'] = change > 5.0
            else:
                skipped.append((row['id'], main_ticker))
        except Exception as e:
            print(f"[⚠️] Row error ({row['id']}): {e}")
            skipped.append((row['id'], 'Error'))

    df_cleaned = df.dropna(subset=['price_change'])
    df_cleaned.to_csv(output_csv, index=False)

    print(f"\n✅ Cleaned data saved to {output_csv}")
    print(f"❌ Skipped {len(skipped)} posts due to invalid tickers or missing price data")
    if skipped:
        print("Examples:", skipped[:5])

# Run it
if __name__ == "__main__":
    enrich_data("reddit_posts_large.csv", "reddit_posts_labeled_clean.csv")
