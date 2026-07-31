import pandas as pd
import string

df = pd.read_csv("app/data/nysc_faq.csv")


def preprocess_text(text):
    text = text.lower()


    text = text.translate(str.maketrans("", "", string.punctuation))

    return text


df["Cleaned_Question"] = df["Question"].apply(preprocess_text)

print(df[["Question", "Cleaned_Question"]])