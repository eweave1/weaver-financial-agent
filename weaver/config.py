"""Configuration loading from config.yaml and environment variables."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

load_dotenv()

# Default: config.yaml at project root (two levels up from this file)
_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load and return configuration from a YAML file.

    Resolution order:
    1. Explicit config_path argument
    2. WEAVER_CONFIG_PATH environment variable
    3. config.yaml at the project root
    """
    if config_path:
        path = config_path
    elif env_path := os.getenv("WEAVER_CONFIG_PATH"):
        path = Path(env_path)
    else:
        path = _DEFAULT_CONFIG_PATH

    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_vault_path(config: dict[str, Any]) -> Path:
    """Return the Obsidian vault path.

    Resolution order:
    1. OBSIDIAN_VAULT_PATH environment variable
    2. vault_path key in config.yaml

    Raises ValueError if neither is set.
    """
    env_path = os.getenv("OBSIDIAN_VAULT_PATH")
    if env_path:
        return Path(env_path)

    config_path = config.get("vault_path")
    if config_path:
        return Path(config_path)

    raise ValueError(
        "Vault path not configured. "
        "Set OBSIDIAN_VAULT_PATH in .env, or set vault_path in config.yaml."
    )


def get_watchlist(config: dict[str, Any]) -> list[str]:
    """Return a flat list of all tickers from the watchlist."""
    watchlist = config.get("watchlist", {})
    tickers: list[str] = []
    for group in watchlist.values():
        tickers.extend(group or [])
    return tickers


def get_analyze_model(config: dict[str, Any]) -> str:
    """Return the AI model slug for wf research --analyze."""
    return config.get("ai", {}).get("analyze_model", "deepseek/deepseek-v4-pro")


def get_openrouter_timeout(config: dict[str, Any]) -> int:
    """Return the OpenRouter request timeout in seconds."""
    return int(config.get("ai", {}).get("openrouter_timeout_seconds", 90))
