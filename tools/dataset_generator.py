import pandas as pd
import os

print("=" * 50)
print("NYSC DATASET GENERATOR")
print("=" * 50)

categories = [
    "registration",
    "camp",
    "mobilization",
    "relocation",
    "exemption",
    "foreign_graduates",
    "payment",
    "certificate",
    "posting",
    "clearance"
]

print("\nAvailable Categories:")
for i, cat in enumerate(categories, 1):
    print(f"{i}. {cat}")

choice = int(input("\nChoose category number: "))
category = categories[choice - 1]

question = input("Question: ").strip()
answer = input("Answer: ").strip()

file_path = f"app/data/{category}.csv"

if os.path.exists(file_path):
    df = pd.read_csv(file_path)
else:
    df = pd.DataFrame(columns=["Question", "Answer", "Category"])

new_row = pd.DataFrame([{
    "Question": question,
    "Answer": answer,
    "Category": category.title()
}])

df = pd.concat([df, new_row], ignore_index=True)

df.to_csv(file_path, index=False)

print("\n✅ FAQ added successfully!")
print(f"Category: {category}")
print(f"Total FAQs: {len(df)}")