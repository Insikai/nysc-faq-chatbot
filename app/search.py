from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class SearchEngine:
    def get_confidence_level(self, score):
        """
        Classify search confidence based on the final similarity score.
        """

        if score >= 0.75:
            return "high"

        if score >= 0.50:
            return "medium"

        return "low"
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

        self.keyword_synonyms = {

            "photo": {
                "photograph",
                "photographs",
                "photos",
                "passport"
            },

            "photos": {
                "photograph",
                "photographs",
                "photo",
                "passport"
            },

            "photograph": {
                "photo",
                "photographs",
                "photos",
                "passport"
            },

            "photographs": {
                "photo",
                "photograph",
                "photos",
                "passport"
            },

            "passport": {
                "photo",
                "photograph",
                "photographs",
                "photos"
            },

            "medical": {
                "certificate",
                "documents",
                "health",
                "healthcare"
            },

            "certificate": {
                "medical",
                "documents"
            },

            "documents": {
                "document",
                "requirements",
                "papers"
            },

            "document": {
                "documents",
                "requirements",
                "papers"
            },

            "requirements": {
                "required",
                "documents",
                "document"
            },

            "husband": {
                "spouse",
                "partner"
            },

            "wife": {
                "spouse",
                "partner"
            },

            "spouse": {
                "husband",
                "wife",
                "partner"
            },

            "partner": {
                "husband",
                "wife",
                "spouse"
            },

            "relocate": {
                "relocation",
                "transfer",
                "move"
            },

            "relocation": {
                "relocate",
                "transfer",
                "move"
            },

            "transfer": {
                "relocate",
                "relocation",
                "move"
            },

            "move": {
                "relocate",
                "relocation",
                "transfer"
            }
        }
        self.intent_keywords = {
            "Eligibility": {
                "age",
                "eligible",
                "qualified",
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
                "medical",
                "passport"
            },
            "Relocation": {
                "relocate",
                "relocation",
                "transfer",
                "husband",
                "wife",
                "spouse",
                "partner",
                "state",
                "move",
                "service"
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

    def detect_intent(
        self,
        question,
        previous_question=""
    ):

        if (
            previous_question
            and self.is_contextual_follow_up(question)
        ):

            previous_intent = self.detect_intent(
                previous_question
            )

            if previous_intent:
                return previous_intent

        words = set(
            question.split()
        )

        expanded_words = set(
            words
        )

        for word in words:
            if word in self.keyword_synonyms:
                expanded_words.update(
                    self.keyword_synonyms[word]
                )

        scores = {}

        for category, keywords in self.intent_keywords.items():
            matches = expanded_words.intersection(
                keywords
            )

            scores[category] = len(
                matches
            )

        if not scores:
            return None

        best_category = max(
            scores,
            key=scores.get
        )

        if scores[best_category] == 0:
            return None

        return best_category

    def is_contextual_follow_up(self, question):

        follow_up_phrases = [
            "what about",
            "what of",
            "how about",
            "and what about",
            "and what of",
            "how long",
            "how long does it take",
            "how much time",
            "when will it",
            "when can it",
            "what else",
            "how about that",
            "what then",
            "and the",
            "and what",
            "when can i",
            "what documents",
            "what document",
            "which documents",
            "which document",
            "what requirements",
            "what requirement",
        ]

        cleaned_question = question.lower().strip()

        for phrase in follow_up_phrases:
            if cleaned_question.startswith(phrase):
                return True

        follow_up_patterns = [
            "what are the requirements",
            "what are requirements",
            "what are the documents",
            "what documents do i need",
            "what documents are required",
            "how much does it cost",
            "how much will it cost",
            "how long does it take",
            "how long will it take",
            "what about for",
            "and for",
        ]

        for pattern in follow_up_patterns:
            if pattern in cleaned_question:
                return True

        return False

    def search(
        self,
        question,
        previous_question=""
    ):
        is_follow_up = bool(
            previous_question
            and self.is_contextual_follow_up(question)
        )
        cleaned = self.preprocess(question)

        similarity = self.vectorizer.transform([cleaned])
        similarity = cosine_similarity(similarity, self.X)[0]

        if is_follow_up and previous_question:
            context_cleaned = self.preprocess(previous_question)
            context_vector = self.vectorizer.transform([context_cleaned])
            context_similarity = cosine_similarity(
                context_vector,
                self.X
            )[0]
            similarity = (
                0.7 * similarity
                + 0.3 * context_similarity
            )

        user_words = set(cleaned.split())
        expanded_words = set(user_words)

        for word in user_words:
            if word in self.keyword_synonyms:
                expanded_words.update(self.keyword_synonyms[word])

        keyword_scores = []

        for faq_question in self.df["Cleaned_Question"]:
            faq_words = set(faq_question.split())

            if not expanded_words:
                keyword_scores.append(0.0)
                continue

            common_words = expanded_words.intersection(faq_words)
            keyword_score = len(common_words) / len(expanded_words)
            keyword_scores.append(keyword_score)

        keyword_scores = np.array(keyword_scores)

        final_scores = (
            0.7 * similarity
            + 0.3 * keyword_scores
        )

        is_contextual_intent = bool(
            previous_question
            and self.is_contextual_follow_up(question)
        )

        if is_contextual_intent:
            context_category = self.detect_intent(
                self.preprocess(previous_question)
            )
        else:
            context_category = None

        intent = self.detect_intent(cleaned)

        if context_category:
            for i, category in enumerate(self.df["Category"]):
                if category == context_category:
                    final_scores[i] += 0.30

        if intent:
            for i, category in enumerate(self.df["Category"]):
                if category == intent:
                    if is_contextual_intent:
                        final_scores[i] += 0.35
                    else:
                        final_scores[i] += 0.10

        final_scores = np.clip(final_scores, 0.0, 1.0)

        top_matches = final_scores.argsort()[-3:][::-1]
        best_match = top_matches[0]
        best_score = final_scores[best_match]

        return (
            top_matches,
            best_match,
            best_score,
            final_scores
        )