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

        self.intent_keywords = {

            "Eligibility": {
                "age",
                "eligible",
                "qualification",
                "qualify",
                "eligibility"
            },

            "Camp": {
                "camp",
                "documents",
                "callup",
                "call",
                "photograph",
                "medical"
            },

            "Relocation": {
                "relocate",
                "relocation",
                "transfer",
                "husband",
                "state"
            },

            "Registration": {
                "register",
                "registration",
                "dashboard",
                "login",
                "portal"
            },

            "Exclusion": {
                "exemption",
                "exempt",
                "excluded",
                "doesnt",
                "not",
                "required",
                "serve"
            },

            "Office": {
                "office",
                "headquarters",
                "address",
                "contact",
                "phone",
                "email"
            }
        }


    def detect_intent(self, question):

        words = set(question.split())

        scores = {}

        for category, keywords in self.intent_keywords.items():

            matches = words.intersection(keywords)

            scores[category] = len(matches)

        if not scores:
            return None

        best_category = max(
            scores,
            key=scores.get
        )

        if scores[best_category] == 0:
            return None

        return best_category


    def search(self, question, previous_question=""):

        # Use the current question as the main search query.
        # Add the previous question as conversational context.

        if previous_question:

            combined_question = (
                question + " " +
                question + " " +
                previous_question
            )

        else:

            combined_question = question


        cleaned = self.preprocess(
            combined_question
        )


        user_vector = self.vectorizer.transform(
            [cleaned]
        )


        similarity = cosine_similarity(
            user_vector,
            self.X
        )[0]


        user_words = set(
            cleaned.split()
        )


        keyword_scores = []


        for faq_question in self.df["Cleaned_Question"]:

            faq_words = set(
                faq_question.split()
            )


            if not user_words:

                keyword_scores.append(0.0)

                continue


            common_words = user_words.intersection(
                faq_words
            )


            keyword_score = (
                len(common_words)
                / len(user_words)
            )


            keyword_scores.append(
                keyword_score
            )


        keyword_scores = np.array(
            keyword_scores
        )


        final_scores = (
            0.7 * similarity
            + 0.3 * keyword_scores
        )


        intent = self.detect_intent(
            cleaned
        )


        if intent:

            for i, category in enumerate(
                self.df["Category"]
            ):

                if category == intent:

                    final_scores[i] += 0.10


        final_scores = np.clip(
            final_scores,
            0.0,
            1.0
        )


        top_matches = (
            final_scores
            .argsort()[-3:][::-1]
        )


        best_match = top_matches[0]


        best_score = final_scores[
            best_match
        ]


        return (
            top_matches,
            best_match,
            best_score,
            final_scores
        )