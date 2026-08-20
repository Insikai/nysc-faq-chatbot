from flask import Flask, jsonify, render_template, request

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

    follow_up_patterns = {
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
    }

    for pattern in follow_up_patterns:
        if pattern in cleaned:
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
def get_recent_history(
    conversation_history,
    current_question="",
    limit=3
):
    """
    Return recent meaningful conversation questions.

    The most recent question is given priority.
    Very low-confidence/unknown questions are ignored
    when possible.
    """

    if not isinstance(
        conversation_history,
        list
    ):
        return []

    current_cleaned = preprocess_text(
        current_question
    ).strip()

    recent_history = []
    seen_questions = set()

    for item in reversed(
        conversation_history
    ):

        if isinstance(item, dict):

            question = item.get(
                "question",
                ""
            ).strip()

            answer = item.get(
                "answer",
                ""
            ).strip()

        elif isinstance(item, str):

            question = item.strip()
            answer = ""

        else:

            continue

        if not question:
            continue

        cleaned_question = preprocess_text(
            question
        ).strip()

        if not cleaned_question:
            continue

        if cleaned_question == current_cleaned:
            continue

        if cleaned_question in seen_questions:
            continue

        if (
            question.lower().strip()
            in {
                "xyzabc123",
                "test",
                "testing"
            }
        ):
            continue

        if isinstance(item, dict) and not answer:
            continue

        seen_questions.add(
            cleaned_question
        )

        recent_history.append(
            question
        )

        if len(recent_history) >= limit:
            break

    return recent_history
    """
    Return recent unique conversation questions,
    excluding the current question.
    """

    if not isinstance(
        conversation_history,
        list
    ):
        return []

    current_cleaned = preprocess_text(
        current_question
    ).strip()

    recent_history = []
    seen_questions = set()

    for item in reversed(
        conversation_history
    ):

        if isinstance(item, dict):

            question = item.get(
                "question",
                ""
            ).strip()

        elif isinstance(item, str):

            question = item.strip()

        else:

            continue

        if not question:
            continue

        cleaned_question = preprocess_text(
            question
        ).strip()

        if not cleaned_question:
            continue

        if cleaned_question == current_cleaned:
            continue

        if cleaned_question in seen_questions:
            continue

        seen_questions.add(
            cleaned_question
        )

        recent_history.insert(
            0,
            question
        )

        if len(recent_history) >= limit:
            break

    return recent_history
@app.route("/")
def home():

    return render_template(
        "index.html"
    )

@app.route("/ask", methods=["POST"])
def ask():

    try:
        data = request.get_json()
    except Exception:
        return jsonify({
            "error": "Invalid JSON request."
        }), 400

    if data is None:
        return jsonify({
            "error": "Invalid JSON request."
        }), 400

    if not isinstance(data, dict):
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

    recent_history = get_recent_history(
        conversation_history,
        current_question=question,
        limit=3
    )
    context_question = ""

    if is_follow_up(question):

        context_parts = []

        if recent_history:
            context_parts.append(
                recent_history[0]
            )
        elif previous_question:
            context_parts.append(
                previous_question
            )

        if context_parts:
            context_question = (
                " ".join(context_parts)
                + " "
                + question
            )

    search_query = context_question if context_question else question

    print("QUESTION:", question)
    print("PREVIOUS QUESTION:", previous_question)
    print("RECENT HISTORY:", recent_history)
    print("CONTEXT QUESTION:", search_query)

    context_for_search = (
        recent_history[0]
        if recent_history
        else previous_question
    )

    top_matches, best_match, best_score, scores = (
        search_engine.search(
            question,
            context_for_search
        )
    )

    confidence = search_engine.get_confidence_level(
        best_score
    )

    print(
        "DEBUG confidence:",
        confidence
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
        search_engine.preprocess(question),
        search_query
    )

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

    return jsonify({

        "answer": df.iloc[best_match]["Answer"],
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
if __name__ == "__main__":
    app.run(
        debug=True
    )