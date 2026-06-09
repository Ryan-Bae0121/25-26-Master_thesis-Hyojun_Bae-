"""Job Events."""

import logging
from datetime import datetime
from typing import Annotated
from uuid import UUID

from pydantic import AliasChoices, Field
from pytz import UTC

from imecilabt.gpulab.schemas.base import AwareDatetime, BaseSchema
from imecilabt.gpulab.schemas.job2 import JobStatus as Job2Status
from imecilabt.gpulab.util.enum import CaseInsensitiveEnum

from .job2 import JobId


class JobEventType(CaseInsensitiveEnum):
    """Job Event Type."""

    STATUS_CHANGE = "Status Change"
    DEBUG = "Debug"
    INFO = "Info"
    WARN = "Warning"
    ERROR = "Error"

    @classmethod
    def from_logging_level(cls, level: int) -> "JobEventType":
        """Get logging level for a given JobEventType."""
        __levelToJobEventType = {
            logging.CRITICAL: JobEventType.ERROR,
            logging.ERROR: JobEventType.ERROR,
            logging.WARNING: JobEventType.WARN,
            logging.INFO: JobEventType.INFO,
            logging.DEBUG: JobEventType.DEBUG,
        }

        return __levelToJobEventType.get(level, JobEventType.DEBUG)


class JobEvent(BaseSchema):
    """Job Event."""

    job_id: Annotated[JobId, Field(validation_alias=AliasChoices("jobId", "job_id"))]

    @property
    def job_uuid(self) -> UUID:
        """Return job UUID."""
        return UUID(self.job_id)

    type: JobEventType
    time: AwareDatetime | None = Field(default_factory=lambda: datetime.now(tz=UTC))
    # backward incompatible: in v4, this should actually be snake case new_state, not camelcase newState.
    #                        We could fix it, but it's very ugly or not forward compatible,
    #                        So we'll break it. The site and CLI should not error on this, just not show the info.
    new_state: Annotated[
        Job2Status | None, Field(validation_alias=AliasChoices("newState", "new_state", "new_status"))
    ] = None
    msg: str | None = None

    def __str__(self) -> str:
        """Return as string."""
        if self.type == JobEventType.STATUS_CHANGE:
            return (
                "JobEvent(STATUS_CHANGE, "
                f"{self.new_state.name if self.new_state else None}, "
                f"{self.job_id}, "
                f"{self.time.isoformat() if self.time else None})"
            )
        return (
            f"JobEvent({self.type.name if self.type else None}, "
            f"{self.msg}, "
            f"{self.job_id}, "
            f"{self.time.isoformat() if self.time else None})"
        )
