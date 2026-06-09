"""Enum utils."""

from enum import Enum
from typing import Any, TypeVar

from pydantic import GetCoreSchemaHandler, GetJsonSchemaHandler
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import CoreSchema, core_schema

CIEType = TypeVar("CIEType", bound="CaseInsensitiveEnum")


class CaseInsensitiveEnum(Enum):
    """Case Insensitive Enum."""

    @classmethod
    def _missing_(cls: type[CIEType], value: object) -> CIEType | None:
        """Lookup value not found in enum."""
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            value = value.lower()
            for member in cls:
                if member.name.lower() == value:
                    return member
        return None

    @classmethod
    def _validate(cls, v: Any) -> Any:
        try:
            if v in cls:
                return v
        except TypeError:  # pragma: no cover
            pass
        # not a member...look up by name
        try:
            return cls(v)
        except ValueError as err:
            keys = ",".join(f"'{key}'" for key in cls.__members__)
            msg = f"Input should be one of: {keys}"
            raise ValueError(msg) from err

    @classmethod
    def _serialize(cls, v: Any) -> Any:
        if isinstance(v, cls):
            return v.name
        return v

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: GetCoreSchemaHandler) -> CoreSchema:
        """Get Pydantic Core Schema."""
        schema = handler(source_type)

        return core_schema.no_info_before_validator_function(
            cls._validate,
            schema=schema,
            serialization=core_schema.plain_serializer_function_ser_schema(cls._serialize),
        )

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema: CoreSchema, handler: GetJsonSchemaHandler) -> JsonSchemaValue:
        """Get Pydantic JSON Schema."""
        json_schema = handler(core_schema)
        json_schema = handler.resolve_ref_schema(json_schema)
        json_schema["enum"] = list(cls.__members__)
        json_schema["type"] = "str"
        return json_schema
