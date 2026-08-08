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

    matches = []

    for i in top_matches:
        matches.append({
            "question": df.iloc[i]["Question"],
            "score": round(float(scores[i]), 2)
        })

    if best_score < 0.70:
        return jsonify({
            "answer": "Sorry, I don't have an exact answer. Please try one of the suggested questions.",
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