"""Pydantic validators."""

from typing import Any

from pydantic import BeforeValidator


def _ensure_as_list(v: Any) -> list[Any]:
    if v is None:
        return []
    if not isinstance(v, list):
        return [v]
    return v


EnsureListValidator = BeforeValidator(_ensure_as_list)
