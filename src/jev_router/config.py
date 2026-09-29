"""Environment loading. Never logs or echoes secret values."""

import os
from collections.abc import Mapping
from pathlib import Path

KEY_NAME = "JEV_API_KEY"


class ConfigError(RuntimeError):
    pass


def parse_env(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key.strip()] = value
    return values


def load_api_key(
    env_path: Path | str = ".env",
    environ: Mapping[str, str] | None = None,
    name: str = KEY_NAME,
) -> str:
    environ = os.environ if environ is None else environ
    if environ.get(name):
        return environ[name]
    path = Path(env_path)
    if path.exists():
        value = parse_env(path.read_text()).get(name)
        if value:
            return value
    raise ConfigError(f"{name} not found in environment or {path}")
