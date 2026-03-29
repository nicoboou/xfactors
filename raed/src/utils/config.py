from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def _cast(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    if value.startswith("[") and value.endswith("]"):
        raw_items = value[1:-1].strip()
        if not raw_items:
            return []
        return [_cast(item.strip()) for item in raw_items.split(",")]
    return value


def load_config(config_path: str) -> dict[str, Any]:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid config at {config_path}: top-level object must be a mapping")
    return cfg


def apply_overrides(cfg: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    out = deepcopy(cfg)
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override '{override}', expected key=value")
        key, raw_value = override.split("=", 1)
        value = _cast(raw_value)
        parts = key.split(".")
        cursor: dict[str, Any] = out
        for part in parts[:-1]:
            if part not in cursor or not isinstance(cursor[part], dict):
                cursor[part] = {}
            cursor = cursor[part]
        cursor[parts[-1]] = value
    return out
