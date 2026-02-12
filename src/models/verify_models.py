from pathlib import Path


REQUIRED_FILES = [
    "models/intent_classifier.pkl",
    "models/vectorizer.pkl",
]


def check_models():
    project_root = Path(__file__).resolve().parents[2]
    missing = []
    empty = []

    for rel_path in REQUIRED_FILES:
        path = (project_root / rel_path).resolve()
        if not path.exists():
            missing.append(path)
            continue
        if path.stat().st_size == 0:
            empty.append(path)

    return missing, empty


def main() -> None:
    missing, empty = check_models()

    if not missing and not empty:
        print("All required model files exist and are non-empty.")
        return

    if missing:
        print("Missing model files:")
        for path in missing:
            print(f" - {path}")

    if empty:
        print("Empty model files:")
        for path in empty:
            print(f" - {path}")

    print("\nRegenerate ML models with:")
    print("  python -m src.models.train_ml_models")


if __name__ == "__main__":
    main()
