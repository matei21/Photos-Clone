from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent


def resolve_project_path(relative_path: str) -> Path:
    path = Path(relative_path)
    if path.is_absolute():
        return path

    candidates = [BACKEND_DIR / path, PROJECT_ROOT / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]
