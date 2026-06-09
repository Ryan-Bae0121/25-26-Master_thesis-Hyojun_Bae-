"""Job Filter."""

from collections.abc import Sequence
from typing import Annotated, Any

from pydantic import (
    AfterValidator,
    BeforeValidator,
    ConfigDict,
    Field,
    PlainSerializer,
    ValidationInfo,
)

from imecilabt.gpulab.schemas.base import BaseSchema
from imecilabt.gpulab.schemas.job2 import JobStatus
from imecilabt.gpulab.schemas.slave_info2 import SlaveInfo2


def _validate_allowed_states(allowed_states: list[JobStatus], info: ValidationInfo) -> list[JobStatus]:
    if len(allowed_states):
        for state in ["pending", "finished", "running", "deleted"]:
            if info.data.get(f"{state}_state"):
                msg = f"Cannot combine {state} and allowed_states filters."
                raise ValueError(msg)

        return allowed_states

    if info.data.get("pending_state"):
        allowed_states.extend((JobStatus.ONHOLD, JobStatus.QUEUED))
    if info.data.get("finished_state"):
        allowed_states.extend(
            (
                JobStatus.FINISHED,
                JobStatus.FAILED,
                JobStatus.CANCELLED,
                JobStatus.HALTED,
            )
        )
    if info.data.get("running_state"):
        allowed_states.extend(
            (
                JobStatus.RUNNING,
                JobStatus.STARTING,
                JobStatus.MUSTHALT,
                JobStatus.HALTING,
            )
        )
    if info.data.get("deleted_state"):
        allowed_states.append(JobStatus.DELETED)

    if not allowed_states:
        # If nothing in allowed_states:  show all states, except delete
        allowed_states = [
            JobStatus.ONHOLD,
            JobStatus.QUEUED,
            JobStatus.FINISHED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
            JobStatus.HALTED,
            JobStatus.RUNNING,
            JobStatus.STARTING,
            JobStatus.MUSTHALT,
            JobStatus.HALTING,
        ]

    return allowed_states


def _serialize_allowed_states(allowed_states: list[JobStatus], _info: Any) -> str:
    return ",".join(state.name for state in allowed_states)


def _is_list_int_or_int(value: Any) -> list[int]:
    if not value:
        return []
    if isinstance(value, list):
        for item in value:
            if not isinstance(item, int):
                msg = f"{item} in list is not an int"
                raise TypeError(msg)
        return value
    if not isinstance(value, int):
        msg = f"{value} is not an int"
        raise TypeError(msg)
    return [value]


def _is_list_str_or_str(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        for item in value:
            if not isinstance(item, str):
                msg = f"{item} in list is not a str"
                raise TypeError(msg)
        return value
    if not isinstance(value, str):
        msg = f"{value} is not a str"
        raise TypeError(msg)
    return [value]


class JobFilter3(BaseSchema):
    """Job Filter."""

    cluster_id: Annotated[
        list[int],
        BeforeValidator(_is_list_int_or_int),
    ] = []  # requested OR assigned cluster ID
    # Note: mutable default is OK in pydantic!
    #   https://docs.pydantic.dev/latest/concepts/models/#fields-with-non-hashable-default-values

    pending_state: Annotated[bool | None, Field(exclude=True, alias="pending")] = None
    finished_state: Annotated[bool | None, Field(exclude=True, alias="finished")] = None
    running_state: Annotated[bool | None, Field(exclude=True, alias="running")] = None
    deleted_state: Annotated[bool | None, Field(exclude=True, alias="deleted")] = None

    allowed_states: Annotated[
        list[JobStatus],
        AfterValidator(_validate_allowed_states),
        PlainSerializer(_serialize_allowed_states),
    ] = Field(default_factory=list, validate_default=True)  # [] means: don't filter

    user_id: Annotated[
        list[str],
        BeforeValidator(_is_list_str_or_str),
        Field(validation_alias="userid"),
    ] = []
    user_urn: Annotated[
        list[str],
        BeforeValidator(_is_list_str_or_str),
        Field(validation_alias="userurn"),
    ] = []
    user_name: Annotated[
        list[str],
        BeforeValidator(_is_list_str_or_str),
        Field(validation_alias="username"),
    ] = []
    project_id: Annotated[
        list[str],
        BeforeValidator(_is_list_str_or_str),
        Field(validation_alias="projectid"),
    ] = []
    project_urn: Annotated[
        list[str],
        BeforeValidator(_is_list_str_or_str),
        Field(validation_alias="projecturn"),
    ] = []
    project_name: Annotated[
        list[str],
        BeforeValidator(_is_list_str_or_str),
        Field(validation_alias="projectname"),
    ] = []
    assigned_slave_name: Annotated[list[str], BeforeValidator(_is_list_str_or_str)] = []
    assigned_slave_instance_id: Annotated[list[str], BeforeValidator(_is_list_str_or_str)] = []
    interactive: bool | None = None
    waste: bool | None = None
    reservation_id: str | None = None

    @classmethod
    def for_pending(cls) -> "JobFilter3":
        """Typical JobFilter used to find all Pending Jobs."""
        return cls(
            allowed_states=[JobStatus.QUEUED],
        )

    @classmethod
    def for_running(cls) -> "JobFilter3":
        """Typical JobFilter used to find all Pending Jobs."""
        return cls(
            allowed_states=[
                JobStatus.RUNNING,
                JobStatus.STARTING,
                JobStatus.MUSTHALT,
                JobStatus.HALTING,
            ],
        )

    @classmethod
    def no_filter(cls) -> "JobFilter3":
        """Get all jobs."""
        return cls(allowed_states=[])

    def fix_assigned_slave_name(self, slave_infos: Sequence[SlaveInfo2]) -> "JobFilter3":
        """Update Job filter to use the assigned slave name instead of an alias of the slave."""
        if not slave_infos:
            slave_infos = []
        fixed_assigned_slave_name = [
            slave_info.name
            for slave_info in slave_infos
            for asn in self.assigned_slave_name
            if slave_info.match_name(asn)
        ]

        if not fixed_assigned_slave_name:
            # If there are no matching slave info's at all, do not fix anything. This will cause an empty result.
            fixed_assigned_slave_name = self.assigned_slave_name
            # return self

        return JobFilter3(
            cluster_id=self.cluster_id,
            allowed_states=self.allowed_states,
            user_urn=self.user_urn,
            user_name=self.user_name,
            project_urn=self.project_urn,
            project_name=self.project_name,
            assigned_slave_name=fixed_assigned_slave_name,
            assigned_slave_instance_id=self.assigned_slave_instance_id,
            interactive=self.interactive,
            waste=self.waste,
        )

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
        from_attributes=True,
    )


VALID_SORT_COLUMNS = [
    "project_name",
    "user_name",
    "name",
    "waste",
    "tallyIncrement",
    "status",
    "gpus",
    "cpus",
    "cpuMemoryGb",
    "updated",
    "created",
    "start_date",
    "end_date",
    "cluster_id",
    "runhost",
]


class JobSort(BaseSchema):
    """Job Sort Parameters."""

    column: str
    ascending: bool = True

    @classmethod
    def parse_sort_string(cls, sort: str) -> list["JobSort"]:
        """Parse sort param string into JobSort object."""
        sort_parts = sort.split(",")

        result = []

        for part in sort_parts:
            descending = part[0] in ["+", "-"] and part[0] == "-"
            name = part[1:] if part[0] in ["+", "-"] else part

            if name not in VALID_SORT_COLUMNS:
                msg = f"Unknown column name: {name}"
                raise ValueError(msg)

            result.append(JobSort(column=name, ascending=not descending))

        return result
