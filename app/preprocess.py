import string


def preprocess_text(text):
    """
    Clean user input and dataset questions.
    """
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text