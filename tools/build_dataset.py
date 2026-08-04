import pandas as pd
import os

print("===== NYSC FAQ Dataset Builder =====\n")

category = input(
    "Enter category (registration, camp, relocation, payment, mobilization, certificates, etc.): "
).strip().lower()

file_path = f"app/data/{category}.csv"

question = input("Enter the question: ").strip()
answer = input("Enter the answer: ").strip()

new_data = {
    "Question": question,
    "Answer": answer,
    "Category": category.title()
}
if os.path.exists(file_path):
    df = pd.read_csv(file_path)
else:
    df = pd.DataFrame(columns=["Question", "Answer", "Category"])

if question.lower() in df["Question"].str.lower().values:
    print("\n⚠️ This question already exists!")
else:
    df.loc[len(df)] = new_data

    df = df.sort_values(by="Question").reset_index(drop=True)

    df.to_csv(file_path, index=False)

    print("\n✅ FAQ added successfully!")
    print(f"📁 Saved to: {file_path}")
    print(f"📚 Total FAQs: {len(df)}")