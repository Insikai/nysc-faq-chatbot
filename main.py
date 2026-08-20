import pandas as pd
from app.search import SearchEngine
from app.dataset import load_dataset
from app.preprocess import preprocess_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = load_dataset()


df["Cleaned_Question"] = df["Question"].apply(preprocess_text)

search_engine = SearchEngine(df, preprocess_text)
print("\n===== NYSC FAQ Chatbot =====")
print("Type 'exit' to quit.\n")

while True:
    user_question = input("You: ")

    if not user_question.strip():
        print("\nChatbot: Please enter a question.\n")
        continue

    if user_question.lower() == "exit":
        print("Chatbot: Goodbye!")
        break

    top_matches, best_match, best_score, similarity = search_engine.search(user_question)

    print(f"\nSimilarity Score: {best_score:.2f}")

    print("\nTop Matches:")
    for rank, i in enumerate(top_matches, start=1):
        score = similarity[i]
        print(f"{rank}. {df.iloc[i]['Question']} ({score:.2f})")

    if best_score < 0.70:
        print("\nChatbot: Sorry, I don't have an exact answer.")
        print("Please try one of the suggested questions above.\n")
    else:
        print("\nCategory:", df.iloc[best_match]["Category"])
        print("Chatbot:", df.iloc[best_match]["Answer"])
        print()