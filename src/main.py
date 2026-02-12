from pathlib import Path
import sys

# Allow running this file directly by ensuring the project root is on sys.path.
if __name__ == "__main__" and __package__ is None:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils import get_logger, check_dl_data, check_model_dir

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = _PROJECT_ROOT / "data" / "dl_training_data.pt"
MODEL_PATH = _PROJECT_ROOT / "models"


def main():
    logger = get_logger("CROSS_CHECK")
    logger.info("Starting cross-check...")

    # Check DL data
    check_dl_data(DATA_PATH, logger)

    # Check model files
    check_model_dir(MODEL_PATH)
    logger.info("Model directory verified")

    logger.info("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
