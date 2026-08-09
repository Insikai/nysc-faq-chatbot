import string


SYNONYMS = {
    "maximum age": "age",
    "age limit": "age",
    "age requirement": "age",
    "how old": "age",
    "old can i be": "age",
    "serve nysc": "nysc service",
    "serving nysc": "nysc service",

    "what should i take to nysc camp": "camp documents",
    "what should i take to camp": "camp documents",
    "what should i bring to nysc camp": "camp documents",
    "what should i bring to camp": "camp documents",

    "what should i take": "camp documents",
    "what should i bring": "camp documents",
    "things to take": "camp documents",
    "things to bring": "camp documents",
    "bring to camp": "camp documents",
    "take to camp": "camp documents",

    "what documents": "documents",
    "documents required": "documents",
    "required documents": "documents",
    "documents do i need": "documents",
    "what do i need": "documents",
}


def preprocess_text(text):
    """
    Clean and normalize user input and dataset questions.
    """

    text = text.lower()

    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    for phrase in sorted(SYNONYMS, key=len, reverse=True):
        text = text.replace(
            phrase,
            SYNONYMS[phrase]
        )

    return text