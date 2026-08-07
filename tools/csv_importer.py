import pandas as pd
import os

print("=" * 50)
print("NYSC CSV IMPORTER")
print("=" * 50)

filename = input("CSV file to import: ").strip()

if not os.path.exists(filename):
    print("File not found.")
    exit()

new_df = pd.read_csv(filename)

new_df = new_df.dropna(subset=["Question", "Answer", "Category"])

required = {"Question", "Answer", "Category"}

if not required.issubset(new_df.columns):
    print("CSV must contain Question, Answer and Category columns.")
    exit()

print(f"\nFound {len(new_df)} FAQs.")

for category in new_df["Category"].unique():

    category_df = new_df[new_df["Category"] == category]

    output = f"app/data/{category.lower().replace(' ', '_')}.csv"

    if os.path.exists(output):
        old_df = pd.read_csv(output)
        merged = pd.concat([old_df, category_df], ignore_index=True)
        merged = merged.drop_duplicates(subset=["Question"])
    else:
        merged = category_df

    merged.to_csv(output, index=False)

    print(f"✔ {category}: {len(category_df)} imported")

print("\nImport completed successfully!")