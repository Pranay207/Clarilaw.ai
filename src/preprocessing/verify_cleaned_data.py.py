import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
file_path = os.path.join(BASE_DIR, "data", "processed", "cleaned_legal_texts.csv")

df = pd.read_csv(file_path)

print("Total rows:", len(df))
print("Empty rows:", df["cleaned_text"].isna().sum())
print("Blank rows:", (df["cleaned_text"].str.strip() == "").sum())
keywords = ["section", "court", "act", "law", "offence", "punishment"]

print("\nLegal Keyword Check:")
for word in keywords:
    count = df["cleaned_text"].str.contains(word, na=False).sum()
    print(f"{word}: {count}")
import re

specials = df["cleaned_text"].apply(
    lambda x: bool(re.search(r"[^a-z\s]", str(x)))
)

print("\nSpecial character rows:", specials.sum())
print("\nSample rows with special characters:")
print(df[specials].head(5))
print("\nSample cleaned texts:")
print(df["cleaned_text"].head(10))
print("\nSource File Distribution:")
print(df["source_file"].value_counts())
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# Test the clean_text function
sample_text = "Section 123: The Act of 2020! @#$$"
cleaned_sample = clean_text(sample_text)
print("\nSample Text Cleaning:")
print("Original:", sample_text)
print("Cleaned:", cleaned_sample)
