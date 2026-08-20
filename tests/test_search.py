from app.dataset import load_dataset
from app.preprocess import preprocess_text
from app.search import SearchEngine


df = load_dataset()

df["Cleaned_Question"] = df["Question"].apply(
    preprocess_text
)

search_engine = SearchEngine(
    df,
    preprocess_text
)


def test_relocation_question():
    results = search_engine.search(
        "Can I relocate to another state?"
    )

    top_matches = results[0]
    best_match = results[1]

    category = df.iloc[best_match]["Category"]

    assert category == "Relocation"


def test_relocation_marriage_follow_up():
    results = search_engine.search(
        "What about for marriage?",
        "Can I relocate to another state?"
    )

    best_match = results[1]

    category = df.iloc[best_match]["Category"]

    assert category == "Relocation"


def test_camp_question():
    results = search_engine.search(
        "What do I need for camp?"
    )

    best_match = results[1]

    category = df.iloc[best_match]["Category"]

    assert category == "Camp"


def test_camp_photograph_follow_up():
    results = search_engine.search(
        "What about photographs?",
        "What do I need for camp?"
    )

    best_match = results[1]

    category = df.iloc[best_match]["Category"]

    assert category == "Camp"


def test_headquarters_question():
    results = search_engine.search(
        "Where is NYSC headquarters?"
    )

    best_match = results[1]

    answer = df.iloc[best_match]["Answer"]

    assert "Abuja" in answer


def test_unrelated_question_does_not_use_old_context():
    results = search_engine.search(
        "Where is NYSC headquarters?",
        "Can I relocate to another state?"
    )

    best_match = results[1]

    answer = df.iloc[best_match]["Answer"]

    assert "Abuja" in answer


def test_relocation_then_marriage_follow_up():
    results = search_engine.search(
        "Can I relocate to another state?",
        ""
    )

    previous_question = df.iloc[results[1]]["Question"]

    results = search_engine.search(
        "What about for marriage?",
        previous_question
    )

    best_match = results[1]

    category = df.iloc[best_match]["Category"]

    assert category == "Relocation"


def test_camp_then_photographs_follow_up():
    results = search_engine.search(
        "What do I need for camp?",
        ""
    )

    previous_question = df.iloc[results[1]]["Question"]

    results = search_engine.search(
        "What about photographs?",
        previous_question
    )

    best_match = results[1]

    category = df.iloc[best_match]["Category"]

    assert category == "Camp"


def test_relocation_marriage_follow_up_has_reasonable_confidence():
    results = search_engine.search(
        "What about for marriage?",
        "Can I relocate to another state?"
    )

    best_score = results[2]

    assert best_score >= 0.50


def test_unrelated_what_about_question_does_not_force_previous_intent():
    results = search_engine.search(
        "What about camp?",
        "Can I relocate to another state?"
    )

    best_match = results[1]

    category = df.iloc[best_match]["Category"]

    assert category == "Camp"


def test_unknown_question_has_low_confidence():
    results = search_engine.search(
        "xyzabc123"
    )

    best_score = results[2]

    assert best_score < 0.50


def test_headquarters_has_high_confidence():
    results = search_engine.search(
        "Where is NYSC headquarters?"
    )

    best_score = results[2]

    assert best_score >= 0.75


def test_relocation_question_has_high_confidence():
    results = search_engine.search(
        "Can I relocate to another state?"
    )

    best_score = results[2]

    assert best_score >= 0.75


def test_vague_question_has_low_confidence():
    results = search_engine.search(
        "What do I need?"
    )

    best_score = results[2]

    assert best_score < 0.50


def test_marriage_follow_up_has_reliable_confidence():
    results = search_engine.search(
        "What about for marriage?",
        "Can I relocate to another state?"
    )

    best_match = results[1]
    best_score = results[2]

    category = df.iloc[best_match]["Category"]

    assert category == "Relocation"
    assert best_score >= 0.50


def test_unrelated_what_about_question_stays_in_new_topic():
    results = search_engine.search(
        "What about camp?",
        "Can I relocate to another state?"
    )

    best_match = results[1]

    category = df.iloc[best_match]["Category"]

    assert category == "Camp"
def test_low_confidence_question_returns_fallback():
    from app.web import app

    client = app.test_client()

    response = client.post(
        "/ask",
        json={
            "question": "How do I do it?"
        }
    )

    data = response.get_json()

    assert response.status_code == 200
    assert data["confidence"] == "low"
    assert data["category"] is None
    assert "not confident enough" in data["answer"].lower()
def test_high_confidence_question_returns_answer():
    from app.web import app

    client = app.test_client()

    response = client.post(
        "/ask",
        json={
            "question": "Where is NYSC headquarters?"
        }
    )

    data = response.get_json()

    assert response.status_code == 200
    assert data["confidence"] == "high"
    assert data["category"] is None
    assert "Abuja" in data["answer"]
def test_empty_question_returns_error():
    from app.web import app

    client = app.test_client()

    response = client.post(
        "/ask",
        json={
            "question": ""
        }
    )

    data = response.get_json()

    assert response.status_code == 400
    assert "Please enter a question." in data["error"]


def test_missing_question_returns_error():
    from app.web import app

    client = app.test_client()

    response = client.post(
        "/ask",
        json={}
    )

    data = response.get_json()

    assert response.status_code == 400
    assert "Please enter a question." in data["error"]


def test_invalid_json_returns_error():
    from app.web import app

    client = app.test_client()

    response = client.post(
        "/ask",
        data="not valid json",
        content_type="application/json"
    )

    data = response.get_json()

    assert response.status_code == 400
    assert "Invalid JSON request." in data["error"]


def test_three_turn_relocation_conversation():
    from app.web import app

    client = app.test_client()

    response = client.post(
        "/ask",
        json={
            "question": "Can I relocate to another state?",
            "conversation_history": []
        }
    )
    assert response.status_code == 200

    response = client.post(
        "/ask",
        json={
            "question": "What about for marriage?",
            "conversation_history": [
                {
                    "question": "Can I relocate to another state?",
                    "answer": response.get_json()["answer"]
                }
            ]
        }
    )

    data = response.get_json()

    assert response.status_code == 200
    assert data["category"] is None
    assert "marriage" in data["answer"].lower() or \
        "relocation" in data["answer"].lower()


def test_recent_context_is_used_for_follow_up():
    from app.web import app

    client = app.test_client()

    response = client.post(
        "/ask",
        json={
            "question": "Can I relocate to another state?",
            "conversation_history": []
        }
    )

    first_answer = response.get_json()["answer"]

    response = client.post(
        "/ask",
        json={
            "question": "What about for marriage?",
            "conversation_history": [
                {
                    "question": "Where is NYSC headquarters?",
                    "answer": "The NYSC headquarters is in Abuja, Nigeria."
                },
                {
                    "question": "Can I relocate to another state?",
                    "answer": first_answer
                }
            ]
        }
    )

    data = response.get_json()

    assert response.status_code == 200
    assert "relocation" in data["answer"].lower() or \
           "marriage" in data["answer"].lower()


def test_unrelated_question_breaks_old_context():
    from app.web import app

    client = app.test_client()

    response = client.post(
        "/ask",
        json={
            "question": "Can I relocate to another state?",
            "conversation_history": []
        }
    )

    relocation_answer = response.get_json()["answer"]

    response = client.post(
        "/ask",
        json={
            "question": "Where is NYSC headquarters?",
            "conversation_history": [
                {
                    "question": "Can I relocate to another state?",
                    "answer": relocation_answer
                }
            ]
        }
    )

    data = response.get_json()

    assert response.status_code == 200
    assert "abuja" in data["answer"].lower()
def test_follow_up_uses_relevant_recent_context():
    from app.web import app

    client = app.test_client()

    response = client.post(
        "/ask",
        json={
            "question": "What about for marriage?",
            "conversation_history": [
                {
                    "question": "Where is NYSC headquarters?",
                    "answer": "The NYSC headquarters is in Abuja, Nigeria."
                },
                {
                    "question": "Can I relocate to another state?",
                    "answer": "Relocation to another state may be possible."
                }
            ]
        }
    )

    data = response.get_json()

    assert response.status_code == 200
    assert (
        "marriage" in data["answer"].lower()
        or "relocation" in data["answer"].lower()
    )


def test_unrelated_recent_question_does_not_replace_relevant_context():
    from app.web import app

    client = app.test_client()

    response = client.post(
        "/ask",
        json={
            "question": "What documents do I need?",
            "conversation_history": [
                {
                    "question": "Can I relocate to another state?",
                    "answer": "Relocation to another state may be possible."
                },
                {
                    "question": "Where is NYSC headquarters?",
                    "answer": "The NYSC headquarters is in Abuja, Nigeria."
                }
            ]
        }
    )

    data = response.get_json()

    assert response.status_code == 200
    assert "document" in data["answer"].lower() or \
           "camp" in data["answer"].lower()


def test_standalone_question_does_not_force_context():
    from app.web import app

    client = app.test_client()

    response = client.post(
        "/ask",
        json={
            "question": "Where is NYSC headquarters?",
            "conversation_history": [
                {
                    "question": "Can I relocate to another state?",
                    "answer": "Relocation to another state may be possible."
                },
                {
                    "question": "What about for marriage?",
                    "answer": "Marriage may be a basis for relocation."
                }
            ]
        }
    )

    data = response.get_json()

    assert response.status_code == 200
    assert "abuja" in data["answer"].lower()