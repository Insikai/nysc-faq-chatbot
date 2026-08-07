from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SearchEngine:
    def __init__(self, dataframe, preprocess):
        self.df = dataframe
        self.preprocess = preprocess

        self.vectorizer = TfidfVectorizer(stop_words="english")

        self.X = self.vectorizer.fit_transform(
            self.df["Cleaned_Question"]
        )

    def search(self, question):

        cleaned = self.preprocess(question)

        user_vector = self.vectorizer.transform([cleaned])

        similarity = cosine_similarity(user_vector, self.X)

        top_matches = similarity[0].argsort()[-3:][::-1]

        best_match = top_matches[0]

        best_score = similarity[0][best_match]

        return top_matches, best_match, best_score, similarity