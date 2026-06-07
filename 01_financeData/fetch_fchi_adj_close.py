from pathlib import Path

import pandas as pd
import yfinance as yf


TICKER = "^FCHI"
START_DATE = "2021-01-01"
END_DATE = "2024-01-01"  # yfinance end date is exclusive
OUTPUT_FILE = Path("FCHI_adj_close_20210101_20231231.csv")


def main() -> None:
    data = yf.download(
        TICKER,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=False,
        progress=False,
    )

    if data.empty:
        raise RuntimeError(f"No data returned for {TICKER}.")

    if isinstance(data.columns, pd.MultiIndex):
        adj_close = data[("Adj Close", TICKER)].to_frame(name="Price")
    else:
        adj_close = data["Adj Close"].to_frame(name="Price")

    adj_close.index.name = "Date"
    adj_close.to_csv(OUTPUT_FILE)

    print(f"Saved {len(adj_close)} rows to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
