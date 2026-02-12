import os
import torch
import numpy as np


def check_file_exists(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"ERROR: File not found: {path}")
    return True


def check_dl_data(data_path: str, logger=None):
    """
    Cross-check DL training data
    """
    check_file_exists(data_path)

    texts, labels = torch.load(data_path, weights_only=False)

    # Allow list OR numpy array
    if isinstance(labels, np.ndarray):
        labels = labels.tolist()

    assert isinstance(texts, list), "Texts must be a list"
    assert isinstance(labels, list), "Labels must be a list"
    assert len(texts) == len(labels), "Texts & labels length mismatch"

    if logger:
        logger.info("DL Data loaded successfully")
        logger.info(f"Samples: {len(texts)}")
        logger.info(f"Unique classes: {len(set(labels))}")

    return texts, labels


def check_model_dir(path: str):
    if not os.path.isdir(path):
        raise FileNotFoundError(f"ERROR: Model directory not found: {path}")

    transformer_files = [
        "config.json",
        "pytorch_model.bin",
        "tokenizer.json",
    ]
    sklearn_files = [
        "intent_classifier.pkl",
        "vectorizer.pkl",
    ]

    transformer_missing = [
        f for f in transformer_files
        if not os.path.exists(os.path.join(path, f))
    ]
    sklearn_missing = [
        f for f in sklearn_files
        if not os.path.exists(os.path.join(path, f))
    ]

    if transformer_missing and sklearn_missing:
        raise FileNotFoundError(
            "ERROR: Missing model files. Expected one of: "
            f"transformer={transformer_files} or sklearn={sklearn_files}. "
            f"Missing transformer={transformer_missing}, missing sklearn={sklearn_missing}"
        )

    return True
