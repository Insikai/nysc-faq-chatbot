import pandas as pd
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


df = pd.read_csv("app/data/nysc_faq.csv")

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

df["Cleaned_Question"] = df["Question"].apply(preprocess_text)

vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(df["Cleaned_Question"])

print("===== NYSC FAQ Chatbot =====")
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

    if best_score < 0.30:
        print("\nChatbot: Sorry, I don't have an answer to that question.\n")
    else:
        print("\nChatbot:", df.iloc[best_match]["Answer"])
        print()