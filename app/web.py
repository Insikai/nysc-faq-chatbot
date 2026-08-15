from flask import Flask, render_template, request, jsonify

from app.dataset import load_dataset
from app.preprocess import preprocess_text
from app.search import SearchEngine


app = Flask(__name__)


df = load_dataset()

df["Cleaned_Question"] = df["Question"].apply(
    preprocess_text
)

search_engine = SearchEngine(
    df,
    preprocess_text
)


def is_follow_up(question):
    """
    Decide whether a question is likely to be a follow-up.
    """

    cleaned = preprocess_text(question).strip()

    if not cleaned:
        return False

    words = cleaned.split()

    follow_up_phrases = {
        "what about",
        "how about",
        "which one",
        "which ones",
        "another state",
        "to another state",
        "that state",
        "the other one",
        "the other",
        "documents",
        "document",
        "requirements",
        "requirement",
        "photos",
        "photographs",
        "medical documents",
        "medical certificate",
        "what documents",
        "what document",
        "which documents",
        "which document",
        "what requirements",
        "what requirement",
        "how long",
        "how long does it take",
        "how much time",
        "when will",
        "when can",
    }

    for phrase in follow_up_phrases:

        if cleaned == phrase:
            return True

        if cleaned.startswith(phrase + " "):
            return True

    if words[0] in {
        "to",
        "for",
        "about",
        "after",
        "before",
        "during",
        "from",
        "with",
        "without"
    }:
        return True

    if len(words) <= 2:

        question_starters = {
            "can",
            "could",
            "would",
            "should",
            "will",
            "what",
            "where",
            "when",
            "who",
            "how",
            "why",
            "is",
            "are",
            "do",
            "does"
        }

        if words[0] not in question_starters:
            return True

    return False


@app.route("/")
def home():

    return render_template(
        "index.html"
    )

@app.route("/ask", methods=["POST"])
def ask():

    data = request.get_json()

    if not data:
        return jsonify({
            "error": "Invalid JSON request."
        }), 400

    question = data.get(
        "question",
        ""
    ).strip()

    previous_question = data.get(
        "previous_question",
        ""
    ).strip()

    conversation_history = data.get(
        "conversation_history",
        []
    )

    if not question:
        return jsonify({
            "error": "Please enter a question."
        }), 400

    # --------------------------------------------------
    # BUILD CONTEXT
    # --------------------------------------------------

    context_question = ""

    if is_follow_up(question) and previous_question:

        context_question = (
            previous_question
            + " "
            + question
        )

    print("QUESTION:", question)
    print("PREVIOUS QUESTION:", previous_question)
    print("CONTEXT QUESTION:", context_question)

    # --------------------------------------------------
    # SEARCH
    # --------------------------------------------------

    top_matches, best_match, best_score, scores = (
        search_engine.search(
            question,
            context_question
        )
    )

    confidence = search_engine.get_confidence_level(
        best_score
    )

    print(
        "DEBUG confidence:",
        confidence
    )

    # --------------------------------------------------
    # BUILD TOP MATCHES
    # --------------------------------------------------

    matches = []

    seen_questions = set()

    for i in top_matches:

        question_text = df.iloc[i]["Question"]

        if question_text in seen_questions:
            continue

        seen_questions.add(
            question_text
        )

        matches.append({

            "question": question_text,

            "score": round(
                float(scores[i]),
                2
            )

        })

        if len(matches) == 3:
            break

    # --------------------------------------------------
    # CONFIDENCE GAP
    # --------------------------------------------------

    if len(matches) > 1:

        second_best_score = matches[1]["score"]

    else:

        second_best_score = 0.0

    confidence_gap = (
        best_score
        - second_best_score
    )

    # --------------------------------------------------
    # INTENT
    # --------------------------------------------------

    intent = search_engine.detect_intent(
        search_engine.preprocess(question),
        context_question
    )

    # --------------------------------------------------
    # LOW CONFIDENCE
    # --------------------------------------------------

    if confidence == "low":

        return jsonify({

            "answer": (
                "I'm not confident enough about "
                "the answer. Please rephrase your "
                "question or choose one of the "
                "suggested questions."
            ),

            "category": None,

            "score": round(
                float(best_score),
                2
            ),

            "confidence": confidence,

            "matches": matches

        })

    # --------------------------------------------------
    # AMBIGUOUS QUESTION
    # --------------------------------------------------

    if (
        confidence_gap < 0.10
        and best_score < 0.90
        and intent is None
    ):

        return jsonify({

            "answer": (
                "Your question could mean "
                "a few different things. "
                "Please choose one of the "
                "suggested questions."
            ),

            "category": None,

            "score": round(
                float(best_score),
                2
            ),

            "confidence": confidence,

            "matches": matches,

            "conversation_context": {

                "last_question": question,

                "history_length": len(
                    conversation_history
                )

            }

        })

    # --------------------------------------------------
    # NORMAL ANSWER
    # --------------------------------------------------

    return jsonify({

        "answer": df.iloc[best_match]["Answer"],

        "category": df.iloc[best_match]["Category"],

        "score": round(
            float(best_score),
            2
        ),

        "confidence": confidence,

        "matches": matches,

        "conversation_context": {

            "last_question": question,

            "history_length": len(
                conversation_history
            )

        }

    })

if __name__ == "__main__":

    app.run(
        debug=True
    )