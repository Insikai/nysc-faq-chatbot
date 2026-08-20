from app.dataset import load_dataset
from app.preprocess import preprocess_text
from app.search import SearchEngine


df = load_dataset()


df["Cleaned_Question"] = df["Question"].apply(
    preprocess_text
)

search_engine = SearchEngine(
    df,
    preprocess_text
)

correct = 0
total = len(df)

for index, row in df.iterrows():

    question = row["Question"]

    top_matches, best_match, score, _ = (
        search_engine.search(question)
    )

    if best_match == index:
        correct += 1


accuracy = correct / total * 100

print()
print("===== NYSC FAQ RETRIEVAL EVALUATION =====")
print(f"FAQs evaluated: {total}")
print(f"Correct retrievals: {correct}")
print(f"Retrieval consistency: {accuracy:.2f}%")
print()