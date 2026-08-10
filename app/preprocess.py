import string

SYNONYMS = {
    "maximum age": "age",
    "age limit": "age",
    "age requirement": "age",
    "how old": "age",
    "old can i be": "age",

    "serve nysc": "nysc service",
    "serving nysc": "nysc service",

    "what should i take": "documents",
    "what should i bring": "documents",
    "things to take": "documents",
    "things to bring": "documents",
    "bring to camp": "camp documents",
    "take to camp": "camp documents",

    "what documents": "documents",
    "documents required": "documents",
    "required documents": "documents",
    "documents do i need": "documents",
    "what do i need": "documents",

    "what should i take to nysc camp": "camp documents",
    "what should i take to camp": "camp documents",
    "what should i bring to nysc camp": "camp documents",
    "what should i bring to camp": "camp documents",

    "doesnt have to": "not required to",
    "dont have to": "not required to",
    "does not have to": "not required to",
    "do not have to": "not required to",
    "not required to": "not required to",

    "what does nysc mean": "what does nysc stand for",
    "what is nysc mean": "what does nysc stand for",
}

def preprocess_text(text):
    """
    Clean and normalize user input and dataset questions.
    """

    text = text.lower()

    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    for phrase, replacement in SYNONYMS.items():
        text = text.replace(phrase, replacement)

    return text