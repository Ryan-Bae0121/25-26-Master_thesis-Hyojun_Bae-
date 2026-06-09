"""Pydantic Serializers."""

from enum import Enum

from pydantic import PlainSerializer


def _enum_to_key(v: Enum) -> str:
    # print(v)
    # print(type(v))
    if isinstance(v, Enum):
        return v.name
    return str(v)


EnumAsKeySerializer = PlainSerializer(_enum_to_key)
