from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from fetch_stoxx50e_adj_close import OUTPUT_FILE

PNG_FILE = Path("STOXX50E_daily_adjusted_close.png")


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    csv_path = OUTPUT_FILE if OUTPUT_FILE.is_absolute() else script_dir / OUTPUT_FILE
    png_path = PNG_FILE if PNG_FILE.is_absolute() else script_dir / PNG_FILE

    data = pd.read_csv(csv_path, parse_dates=["Date"])
    if data.empty:
        raise RuntimeError(f"No data found in {csv_path}.")

    data = data.sort_values("Date")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data["Date"], data["Price"], linewidth=1.2)
    ax.set(
        title="EURO STOXX 50 Daily Adjusted Close",
        xlabel="Date",
        ylabel="Adjusted close",
    )
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    print(f"Saved chart to {png_path}")
    plt.show()


if __name__ == "__main__":
    main()
