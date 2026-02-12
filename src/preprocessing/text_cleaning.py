import re
import pandas as pd
import os

# ----------------------------
# Text Cleaning Function
# ----------------------------
def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ----------------------------
# Safe CSV Reader (handles encoding issues)
# ----------------------------
def read_csv_safely(file_path):
    try:
        return pd.read_csv(file_path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(file_path, encoding="latin1")


# ----------------------------
# Main Execution
# ----------------------------
def main():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    raw_folder = os.path.join(BASE_DIR, "data", "raw")
    output_path = os.path.join(BASE_DIR, "data", "processed", "cleaned_legal_texts.csv")

    os.makedirs(os.path.join(BASE_DIR, "data", "processed"), exist_ok=True)

    cleaned_data = []

    for file in os.listdir(raw_folder):
        if not file.endswith(".csv"):
            continue

        file_path = os.path.join(raw_folder, file)
        df = read_csv_safely(file_path)

        # Improved auto-detection of text column
        text_column = None
        keywords = ["text", "section", "description", "clause", "content", "body"]

        for col in df.columns:
            if any(key in col.lower() for key in keywords):
                text_column = col
                break

        # Fallback: use last column if nothing matched
        if text_column is None:
            text_column = df.columns[-1]
            print(f"Warning: No keyword match in {file}, using fallback column: '{text_column}'")

        # Clean text
        df["cleaned_text"] = df[text_column].apply(clean_text)
        # TEMPORARY DEBUG CHECK (Before vs After)
        
        df["source_file"] = file

        cleaned_data.append(df[["cleaned_text", "source_file"]])

        print(f"Cleaned: {file} | Text column used: '{text_column}'")

    final_df = pd.concat(cleaned_data, ignore_index=True)
    final_df.to_csv(output_path, index=False, encoding="utf-8")

    print("\nAll legal files cleaned successfully!")
    print(f"Output saved at: {output_path}")


if __name__ == "__main__":
    main()

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
print("Cleaned:", cleaned_sample )


from tokenizer import tokenize_text
from stopwords import remove_stopwords
from lemmatizer import lemmatize_tokens

def preprocess_text(text):
    tokens = tokenize_text(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize_tokens(tokens)
    return tokens









































































































































































































































































































































































































































