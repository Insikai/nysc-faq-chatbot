import glob
import pandas as pd


def load_dataset():
    """
    Load all CSV files from app/data.
    Remove accidental header rows loaded as data.
    """

    csv_files = glob.glob("app/data/*.csv")
    dataframes = []

    for file in csv_files:
        try:
            df = pd.read_csv(file)

            if not df.empty:
                dataframes.append(df)

        except pd.errors.EmptyDataError:
            print(f"Skipped empty file: {file}")

    dataset = pd.concat(
        dataframes,
        ignore_index=True
    )

    dataset = dataset[
        dataset["Category"].astype(str).str.strip() != "Category"
    ].reset_index(drop=True)

    print(f"Loaded {len(csv_files)} CSV files.")
    print(f"Total FAQs: {len(dataset)}")

    return dataset