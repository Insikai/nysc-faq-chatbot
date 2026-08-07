import pandas as pd
import glob
import os

print("=" * 50)
print("NYSC DATASET CLEANER")
print("=" * 50)

csv_files = glob.glob("app/data/*.csv")

if not csv_files:
    print("No CSV files found.")
    exit()

for file in csv_files:
    print(f"\nCleaning {os.path.basename(file)}...")

    df = pd.read_csv(file)

    before = len(df)

    df = df.dropna(subset=["Question", "Answer", "Category"])
    df = df.drop_duplicates(subset=["Question"])
    df = df.drop_duplicates(subset=["Answer"])
    df = df.sort_values(by="Question")

    after = len(df)

    df.to_csv(file, index=False)

    print(f"✔ Removed {before - after} rows")
    print(f"✔ Remaining FAQs: {after}")

print("\nDataset cleaned successfully!")