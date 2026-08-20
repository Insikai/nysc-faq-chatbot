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