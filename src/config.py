from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import yaml


ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT_DIR / "config.yaml"


def load_config(config_path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    """Load project configuration from YAML."""
    with Path(config_path).open("r", encoding="utf-8") as fp:
        config = yaml.safe_load(fp) or {}
    return config


def project_path(*parts: str) -> Path:
    """Resolve a path relative to the project root."""
    return ROOT_DIR.joinpath(*parts)


def ensure_dir(path: str | Path) -> Path:
    path_obj = Path(path)
    if not path_obj.is_absolute():
        path_obj = project_path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


if __name__ == "__main__":
    print(load_config())
