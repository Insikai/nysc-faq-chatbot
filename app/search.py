from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class SearchEngine:

    def __init__(self, dataframe, preprocess):

        self.df = dataframe
        self.preprocess = preprocess

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2)
        )

        self.X = self.vectorizer.fit_transform(
            self.df["Cleaned_Question"]
        )

    def search(self, question):

        cleaned = self.preprocess(question)

        user_vector = self.vectorizer.transform([cleaned])

        similarity = cosine_similarity(
            user_vector,
            self.X
        )[0]

        user_words = set(cleaned.split())

        keyword_scores = []

        for faq_question in self.df["Cleaned_Question"]:

            faq_words = set(faq_question.split())

            if not user_words:
                keyword_scores.append(0.0)
                continue

            common_words = user_words.intersection(faq_words)

            keyword_score = len(common_words) / len(user_words)

            keyword_scores.append(keyword_score)

        keyword_scores = np.array(keyword_scores)

        final_scores = (
            0.7 * similarity
            + 0.3 * keyword_scores
        )

        top_matches = final_scores.argsort()[-3:][::-1]

        best_match = top_matches[0]

        best_score = final_scores[best_match]

        return (
            top_matches,
            best_match,
            best_score,
            final_scores
        )