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