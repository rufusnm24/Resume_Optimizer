"""Utilities for working with OpenAI within the resume optimizer."""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional, Tuple

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - dependency is optional for tests
    OpenAI = None  # type: ignore[assignment]

# Central place to manage the preferred lightweight model for tooling tasks.
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"


@lru_cache(maxsize=1)
def get_openai_client() -> Optional["OpenAI"]:
    """Return an OpenAI client if the package and API key are available."""
    if OpenAI is None:
        return None

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    try:
        return OpenAI(api_key=api_key)
    except Exception:
        return None


def validate_openai_setup() -> Tuple[bool, Optional[str]]:
    """Check whether OpenAI can be used. Returns (is_available, error_message)."""
    if OpenAI is None:
        return False, "openai package is not installed"

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return False, "OPENAI_API_KEY environment variable is not set"

    client = get_openai_client()
    if client is None:
        return False, "failed to create OpenAI client"

    try:
        # Lightweight auth check. We do not inspect the entire response for performance.
        client.models.list()
    except Exception as exc:
        return False, f"OpenAI API key validation failed: {exc}"

    return True, None
