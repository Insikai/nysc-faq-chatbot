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


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():

    data = request.get_json()

    question = data.get("question", "").strip()

    if not question:
        return jsonify({
            "error": "Please enter a question."
        }), 400

    top_matches, best_match, best_score, scores = search_engine.search(
        question
    )

    # Build unique suggestions
    matches = []
    seen_questions = set()

    for i in top_matches:

        question_text = df.iloc[i]["Question"]

        if question_text in seen_questions:
            continue

        seen_questions.add(question_text)

        matches.append({
            "question": question_text,
            "score": round(float(scores[i]), 2)
        })

        if len(matches) == 3:
            break

    # Use the second unique suggestion for confidence
    if len(matches) > 1:
        second_best_score = matches[1]["score"]
    else:
        second_best_score = 0.0

    confidence_gap = best_score - second_best_score

    # Handle weak matches
    if best_score < 0.70:

        return jsonify({
            "answer": "I'm not confident enough about the answer. Please try one of the suggested questions.",
            "category": None,
            "score": round(float(best_score), 2),
            "matches": matches
        })

    # Handle ambiguous matches
    if confidence_gap < 0.10:

        return jsonify({
            "answer": "Your question could mean a few different things. Please choose one of the suggested questions.",
            "category": None,
            "score": round(float(best_score), 2),
            "matches": matches
        })

    return jsonify({
        "answer": df.iloc[best_match]["Answer"],
        "category": df.iloc[best_match]["Category"],
        "score": round(float(best_score), 2),
        "matches": matches
    })


if __name__ == "__main__":
    app.run(debug=True)