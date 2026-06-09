"""Base Schema for GPULab."""

from datetime import datetime
from typing import Annotated, Any, TypeVar

from pydantic import AwareDatetime as PydanticAwareDatetime
from pydantic import BaseModel, ConfigDict, PlainSerializer
from pydantic.alias_generators import to_camel
from pytz import UTC
from typing_extensions import deprecated


def convert_datetime_to_rfc3339_string(dt: datetime) -> str:
    """Convert datetime to UTC-based RFC3339 string."""
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


AGPULabSchema = TypeVar("AGPULabSchema", bound=BaseModel)


class BaseSchema(BaseModel):
    """Base Schema."""

    model_config = ConfigDict(
        alias_generator=to_camel,
        serialize_by_alias=True,  # We want this. True is not the default.
        validate_by_alias=True,  # We want this. True is already the default.
        # use of camelCase and snake_case was not consistent in the past.
        validate_by_name=True,  # Good for backward compatibility  (was deprecated populate_by_name=True before)
        extra="forbid",
        frozen=True,
        from_attributes=True,
    )

    @classmethod
    @deprecated("Use model_validate_json instead")
    def from_json(
        cls: type[AGPULabSchema],
        json: str,
        *,
        on_unknown_field_override: Any = None,  # noqa: ARG003 ignore for backward compat
    ) -> AGPULabSchema:
        """Parse from JSON."""
        return cls.model_validate_json(json)

    @deprecated("Use model_dump_json(by_alias=True) instead")
    def to_json(self) -> str:
        """Write as JSON."""
        return self.model_dump_json(by_alias=True, exclude_none=True, exclude_unset=True)

    @classmethod
    @deprecated("Use model_validate() instead")
    def from_dict(
        cls: type[AGPULabSchema],
        v: dict[str, Any],
    ) -> AGPULabSchema:
        """Parse from dict."""
        return cls.model_validate(v)

    @deprecated('Use model_dump(mode="json", by_alias=True) instead')
    def to_dict(self) -> dict[str, Any]:
        """Serialize as dict.

        Note, 2 options to replace:
           - model_dump(mode="json", by_alias=True)   -> json compatible dict
           - model_dump(mode="python", by_alias=True) -> dict that might contain datetime and pydantic objects
        """
        return self.model_dump(mode="json", by_alias=True, exclude_none=True, exclude_unset=True)

    @deprecated("Use model_copy instead")
    def make_copy(self: AGPULabSchema) -> AGPULabSchema:
        """Make a copy."""
        return self.model_copy(deep=True)


AwareDatetime = Annotated[
    PydanticAwareDatetime,
    PlainSerializer(convert_datetime_to_rfc3339_string, when_used="json-unless-none"),
]
