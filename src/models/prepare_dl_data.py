import os
from pathlib import Path

import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_legal_texts.csv"
SAVE_PATH = PROJECT_ROOT / "data" / "dl_training_data.pt"
TEXT_COL = "cleaned_text"
LABEL_COL = "category"


def prepare_dl_data():
    df = pd.read_csv(DATA_PATH)
    required_columns = {TEXT_COL, LABEL_COL}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing_display = ", ".join(sorted(missing_columns))
        raise KeyError(
            f"Missing required columns in {DATA_PATH.name}: {missing_display}"
        )

    df = df.dropna(subset=[TEXT_COL, LABEL_COL])

    texts = df[TEXT_COL].astype(str).tolist()
    raw_labels = df[LABEL_COL].astype(str).tolist()

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(raw_labels)

    os.makedirs(SAVE_PATH.parent, exist_ok=True)

    payload = {
        "texts": texts,
        "labels": encoded_labels.tolist(),
        "label_names": label_encoder.classes_.tolist(),
    }
    torch.save(payload, SAVE_PATH)

    print("OK: dl_training_data.pt created successfully")
    print(f"Samples: {len(texts)}")
    print(f"Classes: {len(payload['label_names'])}")
    print(
        "Label mapping:",
        dict(zip(payload["label_names"], range(len(payload["label_names"])))),
    )


if __name__ == "__main__":
    prepare_dl_data()
