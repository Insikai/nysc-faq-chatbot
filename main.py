import pandas as pd
import string
import glob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity 
csv_files = glob.glob("app/data/*.csv")
dataframes = []

for file in csv_files:
    try:
        df = pd.read_csv(file)

        if not df.empty:
            dataframes.append(df)

    except pd.errors.EmptyDataError:
        print(f"Skipped empty file: {file}")

df = pd.concat(dataframes, ignore_index=True)
df = pd.concat(dataframes, ignore_index=True)

print(f"Loaded {len(csv_files)} CSV files.")
print(f"Total FAQs: {len(df)}")

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

df["Cleaned_Question"] = df["Question"].apply(preprocess_text)

vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(df["Cleaned_Question"])

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

    cleaned_question = preprocess_text(user_question)
    user_vector = vectorizer.transform([cleaned_question])

    similarity = cosine_similarity(user_vector, X)

    best_match = similarity.argmax()
    best_score = similarity[0][best_match]

    print(f"\nSimilarity Score: {best_score:.2f}")

    if best_score < 0.70:
        print("\nChatbot: Sorry, I don't have an answer to that question.")

        top_matches = similarity[0].argsort()[-3:][::-1]

        print("\nDid you mean:")
        for i in top_matches:
            print(f"- {df.iloc[i]['Question']}")
        print()

    else:
        print("\nCategory:", df.iloc[best_match]["Category"])
        print("Chatbot:", df.iloc[best_match]["Answer"])
        print()