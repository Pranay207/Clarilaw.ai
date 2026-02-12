import os
from pathlib import Path
import sys

import uvicorn


def _ensure_project_root_on_path() -> None:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))
    os.environ.setdefault("PYTHONPATH", str(project_root))


def main() -> None:
    _ensure_project_root_on_path()
    # Default host/port can be overridden via environment variables.
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("api.main:app", host=host, port=port, reload=True)



if __name__ == "__main__":
    main()

