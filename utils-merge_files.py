from pathlib import Path

import pandas as pd

# Define the folder containing CSV files and the output file name
input_folder = Path(".") / "Output" / "XLNet"
output_file = Path(".") / "Output" / "xlnet-classification.csv"

# List to hold data from all CSV files
csv_list = []

# Loop through all files in the folder
for file in input_folder.iterdir():
    if file.suffix == ".csv":  # Check if the file has a .csv extension
        print(f"Reading {file}...")
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file, sep=";", dtype=str)
        csv_list.append(df)

# Combine all DataFrames into a single DataFrame
combined_df = pd.concat(csv_list, ignore_index=True)

# Save the combined DataFrame to a new CSV file
output_file.parent.mkdir(parents=True, exist_ok=True)
combined_df.to_csv(output_file, index=False, sep=";")
print(f"All files combined into {output_file}.")
