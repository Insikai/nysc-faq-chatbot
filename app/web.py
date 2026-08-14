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


    if not isinstance(
        conversation_history,
        list
    ):

        conversation_history = []


    conversation_history = [
        str(item).strip()
        for item in conversation_history
        if str(item).strip()
    ][-5:]


    context_question = ""


    cleaned_question = preprocess_text(
        question
    ).lower().strip()


    meaningful_keywords = {
        "photograph",
        "photographs",
        "photo",
        "photos",
        "medical",
        "documents",
        "document",
        "relocate",
        "relocation",
        "state",
        "husband",
        "wife",
        "camp"
    }


    has_new_keyword = any(
        keyword in cleaned_question
        for keyword in meaningful_keywords
    )


    context_question = ""

    if is_follow_up(question):

        if previous_question:

            context_question = previous_question

        elif conversation_history:

            context_question = conversation_history[-1]


    print("QUESTION:", question)
    print("PREVIOUS QUESTION:", previous_question)
    print("CONTEXT QUESTION:", context_question)

    top_matches, best_match, best_score, scores = (
        search_engine.search(
            question,
            context_question
        )
    )


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


    if len(matches) > 1:

        second_best_score = matches[1]["score"]

    else:

        second_best_score = 0.0


    confidence_gap = (
        best_score
        - second_best_score
    )

    intent = search_engine.detect_intent(
        preprocess_text(question)
    )

    if (
        best_score < 0.70
        and confidence_gap < 0.15
        and intent is None
    ):
        return jsonify({
            "answer": (
                "I'm not confident enough "
                "about the answer. Please try "
                "one of the suggested questions."
            ),

            "category": None,

            "score": round(
                float(best_score),
                2
            ),

            "matches": matches,

            "conversation_context": {
                "last_question": question,
                "history_length": len(
                    conversation_history
                )
            }

        })



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

            "matches": matches,

            "conversation_context": {
                "last_question": question,
                "history_length": len(
                    conversation_history
                )
            }

        })


    return jsonify({

        "answer": df.iloc[best_match]["Answer"],

        "category": df.iloc[best_match]["Category"],

        "score": round(
            float(best_score),
            2
        ),

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