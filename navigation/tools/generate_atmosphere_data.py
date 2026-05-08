import pandas as pd

from paths import *

INPUT_FILE = DATA_DIR / "atmosphere.xlsx"
OUTPUT_FILE = DATA_DIR / "venus_atmosphere.parquet"


def main():

    df = pd.read_excel(INPUT_FILE)

    # Optional cleanup
    df.columns = [c.strip() for c in df.columns]

    # Ensure sorted by altitude
    df = df.sort_values("altitude")

    # Reset indexing
    df = df.reset_index(drop=True)

    df.to_parquet(OUTPUT_FILE)

    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()