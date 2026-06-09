"""GPULab News."""

from typing import Literal

from pydantic import Field, model_validator

from imecilabt.gpulab.schemas.base import AwareDatetime, BaseSchema


class News(BaseSchema):
    """News Item."""

    id: str = Field(..., examples=["9a6355d0-6d06-11ea-a8da-7b5b9b861935"])
    created: AwareDatetime = Field(..., examples=["2020-03-30T06:38:34Z"])
    enabled: bool = Field(..., examples=[True])
    """Messages can be disabled without deleting them."""
    type: Literal["WARNING", "INFO", "CRITICAL"] = Field(..., examples=["WARNING"])
    title: str = Field(..., examples=["Planned Maintenance Friday Morning"])
    text: str = Field(
        ...,
        examples=["There will be a maintenance friday morning."],
    )
    tags: list[str] = Field(..., examples=[["MAINTENANCE", "WEBSITE", "CLI"]])
    """Flexible systems of tags that can be used to determine where and how to show the messages.

    (Not fixed yet, will be determined by used how this is used.)
    """

    not_before: AwareDatetime | None = Field(None, examples=["2020-03-24T06:38:34Z"])
    not_after: AwareDatetime | None = Field(None, examples=["2020-03-30T06:38:34Z"])

    @model_validator(mode="after")
    def _validate_before_after(self) -> "News":
        if self.not_before and self.not_after and self.not_before > self.not_after:
            msg = "notAfter must come after notBefore"
            raise ValueError(msg)
        return self
