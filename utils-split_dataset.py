from collections.abc import Iterator
from pathlib import Path

import pandas as pd

columns_to_keep = ["patent_id", "patent_title", "patent_abstract"]


def split_large_csv_with_pandas(input_file, output: Path, chunk_size=100_000):
    """
    Splits a large CSV file into smaller files using pandas, each containing chunk_size rows."""

    try:
        chunk_iter: Iterator[pd.DataFrame] = pd.read_csv(
            input_file,
            sep=",",
            dtype=str,
            chunksize=chunk_size,
            iterator=True,
            encoding="ISO-8859-1",
        )

        for i, chunk in enumerate(chunk_iter):
            chunk = chunk[columns_to_keep]
            output_file = output / f"partition_{i}.csv"
            chunk.to_csv(output_file, index=False, sep=";", encoding="utf-8")
            print(f"Written chunk {i} to {output_file}")

        print("Splitting complete.")

    except Exception as e:
        print(f"An error occurred: {e}")


# Example usage
if __name__ == "__main__":
    input_csv_file = Path(".") / "data" / "20240823_Applications_Grants_Combined.csv"
    output_file_prefix = Path(".") / "data" / "20240823_Applications_Grants_Combined"
    output_file_prefix.mkdir(parents=True, exist_ok=True)
    split_large_csv_with_pandas(input_csv_file, output_file_prefix)
