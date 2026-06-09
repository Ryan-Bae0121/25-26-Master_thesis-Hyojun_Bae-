"""GPULab Job Model 2."""

import datetime
import re
from typing import Annotated, Any, Literal, TypeVar, cast
from uuid import UUID

from imecilabt_utils.urn_util import URN
from imecilabt_utils.utils import duration_string_to_seconds
from imecilabt_utils.validation_utils import (
    is_valid_ssh_key,
    is_valid_uuid,
)
from pydantic import (
    AfterValidator,
    AliasChoices,
    BeforeValidator,
    Discriminator,
    EmailStr,
    Field,
    StringConstraints,
    Tag,
    ValidationInfo,
    field_validator,
    model_validator,
)
from typing_extensions import deprecated

from imecilabt.gpulab.schemas.base import AwareDatetime, BaseSchema
from imecilabt.gpulab.schemas.usage_statistics import (
    GPULabUsageStatistics,
    GpuOverview,
    WasteReview,
)
from imecilabt.gpulab.util.convert_utils import (
    urn_to_auth,
    urn_to_name,
    urn_to_user_mini_id,
)
from imecilabt.gpulab.util.enum import CaseInsensitiveEnum
from imecilabt.gpulab.util.validators import EnsureListValidator

DOCKER_IMAGE_PATTERN = (
    r"^((?P<username>[\w\-+_.]+):(?P<password>[^@ ]+)@)?"  # passwords with space or @ are not supported
    r"(?P<repository>[\w.\-_]+(?::\d+|)\/)?"
    r"(?P<image>[a-z0-9.\-_]+(?:\/[a-z0-9.\-_]+|)+)(:(?P<tag>[\w.\-_]{1,127}))?$"
)

DOCKER_IMAGE_REGEX = re.compile(DOCKER_IMAGE_PATTERN)


def validate_job_id(v: str) -> str:
    """Validate Job ID."""
    if not is_valid_uuid(v):
        msg = "Not a valid Job ID."
        raise ValueError(msg)
    return v


def before_validate_job_id(v: Any) -> str:
    """Allow UUID object as JobId.

    This doesn't validate, it just converts UUID to str if needed.
    """
    if v is None:
        msg = "Job ID may not be None."
        raise ValueError(msg)
    if isinstance(v, UUID):
        return str(v)
    if not isinstance(v, str):
        msg = "Invalid type for Job ID."
        raise TypeError(msg)

    return v


JobId = Annotated[str, BeforeValidator(before_validate_job_id), AfterValidator(validate_job_id)]


def validate_reservation_id(v: str) -> str:
    """Validate Reservation ID."""
    if not is_valid_uuid(v):
        msg = "Not a valid reservation ID."
        raise ValueError(msg)
    return v


ReservationId = Annotated[str, AfterValidator(validate_reservation_id)]


class JobStatus(CaseInsensitiveEnum):
    """Job Status."""

    ONHOLD = "On Hold"
    """On hold, not planned to run at this time (not in queue)"""

    QUEUED = "Queued"
    """Available to run, waiting in queue"""
    ASSIGNED = "Assigned"
    """Assigned to a specific node by the scheduler, but not yet "picked up" by that node."""

    STARTING = "Starting"
    """Received by worker, setup in progress, not yet running"""

    RUNNING = "Running"
    """Running on worker"""
    CANCELLED = "Cancelled"
    """Cancelled during run (due to user request)"""
    FINISHED = "Finished"
    """Run completed"""
    DELETED = "Deleted"
    """Marked as deleted. This causes it to be ignored in "queue" view"""
    FAILED = "Failed"
    """Failure due to job definition problem, system problems, or other."""

    # HALT = stopped, but a new job was QUEUED to continue it's work.  (restartable jobs can be halted)
    MUSTHALT = "Must Halt"
    """Master has requested to "halt" the job"""
    HALTING = "Halting"
    """Worker is attempting to "halt" the job"""
    HALTED = "Halted"
    """The job was correctly "halted"""


class HaltReason(CaseInsensitiveEnum):
    """Halt Reason."""

    UNKNOWN = "Unknown"
    USER = "User requested fallback manually"
    ADMIN = "Admin requested fallback"
    SCORE = "Score"
    RESERVATION = "Reservation Priority"
    CLUSTER = "Private Cluster Priority"


class HaltInfo(BaseSchema):
    """Halt Info."""

    reason: HaltReason
    time: datetime.datetime
    started_job_uuids: list[str]
    halted_job_uuids: list[str]
    slave_name: str
    """slave name of the slave the halted jobs run on"""
    slave_instance: str
    """Slave instance the started job(s) will run on.

    Halted jobs might be on this instance, or on stopping instances."""
    started_job_scores: list[float]
    """same order as UUIDs"""
    halted_job_scores: list[float]
    """same order as UUIDs"""


class JobPortMapping(BaseSchema):
    """Job Port Mapping."""

    container_port: int
    host_port: int | None = None
    host_ip: str | None = None

    @classmethod
    def from_docker_dict(cls, d: dict[str, Any], *, container_port: int | str | None = None) -> "JobPortMapping":
        """Make JobPortMapping from the port mapping dict that docker returns.

        :param d: the dict that docker returns
        :param container_port: set the container_port directly instead of using the dict (optional)
        :return:
        """

        def _handle_port_val(val: str | int) -> int:
            if isinstance(val, int):
                return val
            val = str(val)
            val = val.replace("/tcp", "").replace("/udp", "")
            return int(val)

        def _try_handle_port_val(val: str | int | None) -> int | None:
            if val is None:
                return None
            return _handle_port_val(val)

        return cls(
            container_port=_handle_port_val(
                container_port
                if container_port
                else cast("int", d.get("ContainerPort", d.get("containerPort", d.get("container_port"))))
            ),
            host_port=_try_handle_port_val(d.get("HostPort", d.get("hostPort", d.get("host_port")))),
            host_ip=d.get("HostIp", d.get("hostIp", d.get("host_ip"))),
        )


class JobEventTimes(BaseSchema):
    """Job Event Times."""

    created: AwareDatetime
    status_updated: AwareDatetime
    QUEUED: Annotated[AwareDatetime | None, Field(alias="QUEUED")] = None
    ASSIGNED: Annotated[AwareDatetime | None, Field(alias="ASSIGNED")] = None
    STARTING: Annotated[AwareDatetime | None, Field(alias="STARTING")] = None
    RUNNING: Annotated[AwareDatetime | None, Field(alias="RUNNING")] = None
    FINISHED: Annotated[AwareDatetime | None, Field(alias="FINISHED")] = None
    FAILED: Annotated[AwareDatetime | None, Field(alias="FAILED")] = None
    CANCELLED: Annotated[AwareDatetime | None, Field(alias="CANCELLED")] = None
    DELETED: Annotated[AwareDatetime | None, Field(alias="DELETED")] = None
    MUSTHALT: Annotated[AwareDatetime | None, Field(alias="MUSTHALT")] = None
    HALTING: Annotated[AwareDatetime | None, Field(alias="HALTING")] = None
    HALTED: Annotated[AwareDatetime | None, Field(alias="HALTED")] = None
    long_run_notify: AwareDatetime | None = None  # last time long run notify email was sent
    waste_notify: AwareDatetime | None = None  # last time user was notified about job wasting resources

    @property
    def end_date(self) -> AwareDatetime | None:
        """End date."""
        if self.FINISHED:
            return self.FINISHED
        if self.HALTED:
            return self.HALTED
        if self.FAILED:
            return self.FAILED
        if self.CANCELLED:
            return self.CANCELLED
        if self.DELETED:
            return self.DELETED
        return None

    def get_duration(self) -> datetime.timedelta | None:
        """Get current duration of job."""
        start = self.RUNNING
        if start is None:
            return None

        end = self.FINISHED or self.CANCELLED or self.FAILED or self.HALTED or self.DELETED
        if end is None:
            end = datetime.datetime.now(datetime.timezone.utc)
        return end - start

    def get_duration_or_zero(self) -> datetime.timedelta:
        """Get current duration of job."""
        start = self.RUNNING
        if start is None:
            return datetime.timedelta(seconds=0)

        end = self.FINISHED or self.CANCELLED or self.FAILED or self.HALTED or self.DELETED
        if end is None:
            end = datetime.datetime.now(datetime.timezone.utc)
        return end - start

    def get_queue_duration(self) -> datetime.timedelta | None:
        """Get queue duration."""
        start = self.QUEUED
        if start is None:
            return None

        end = self.STARTING or self.RUNNING or self.FAILED or self.HALTED or self.CANCELLED or self.DELETED
        if end is None:
            end = datetime.datetime.now(datetime.timezone.utc)

        return end - start

    def sanitized_copy(self, _logged_in: bool = True, _same_project: bool = False) -> "JobEventTimes":
        """Get anonymized copy."""
        # all may be known
        # TODO maybe hide some more if not logged_in
        return self.model_copy(update={"long_run_notify": None})


class JobStateResources(BaseSchema):
    """Job State Resources."""

    cluster_id: int = Field(..., gt=0)
    gpu_ids: list[int]
    cpu_ids: list[int]
    cpu_memory_gb: int  # does NOT include tmpfs memory (it is listed separately below)
    gpu_memory_gb: int  # if less than a GB, this is rounded DOWN (so it can be 0!)

    slave_name: str
    slave_host: str
    slave_instance_id: str
    slave_instance_pid: int
    worker_id: int

    ssh_host: str | None = None
    ssh_port: int | None = None
    ssh_username: str | None = None
    ssh_proxy_host: str | None = None
    ssh_proxy_port: int | None = None
    ssh_proxy_username: str | None = None

    port_mappings: list[JobPortMapping] = Field(default_factory=list)
    gpu_details: GpuOverview | None = None

    tmpfs_mem_gb: int = Field(0)

    # I don't think we need this.
    # I think this was only used to remove tmpfs_mem_gb from the output. But that doesn't work that way in pydantic.
    # @field_serializer("tmpfs_mem_gb")
    # def _serialize_tmpfs_mem_gb(self, tmpfs_mem_gb: int, _info):
    #     if tmpfs_mem_gb == 0:
    #         return None
    #     return tmpfs_mem_gb

    @field_validator("tmpfs_mem_gb", mode="before")
    @classmethod
    def _validate_tmpfs_mem_gb(cls, v: Any) -> int:
        if v is None:
            return 0
        if not isinstance(v, int):
            msg = "tmpfs_mem_gb must be an int"
            raise TypeError(msg)
        return v

    @property
    def cpu_memory_mb(self) -> int:
        """CPU Memory in MB."""
        # while GB is sometimes 1000^3 instead of 1024^3 (typically HD size etc), for memory, base 1024 is always used.
        # (and docker seems to use 1000 based, which is silly)
        return self.cpu_memory_gb * 1024

    @property
    def cpu_memory_byte(self) -> int:
        """CPU Memory in bytes."""
        # while GB is sometimes 1000^3 instead of 1024^3 (typically HD size etc), for memory, base 1024 is always used.
        # (and docker seems to use 1000 based, which is silly)
        return self.cpu_memory_gb * 1024 * 1024 * 1024

    @property
    def tmpfs_mem_mb(self) -> int:
        """Tmpfs Memory in MB."""
        # while GB is sometimes 1000^3 instead of 1024^3 (typically HD size etc), for memory, base 1024 is always used.
        # (and docker seems to use 1000 based, which is silly)
        return (self.tmpfs_mem_gb or 0) * 1024

    @property
    def tmpfs_mem_byte(self) -> int:
        """Tmpfs Memory in bytes."""
        # while GB is sometimes 1000^3 instead of 1024^3 (typically HD size etc), for memory, base 1024 is always used.
        # (and docker seems to use 1000 based, which is silly)
        return (self.tmpfs_mem_gb or 0) * 1024 * 1024 * 1024

    @property
    def all_cpu_memory_gb(self) -> int:
        """All Memory in GB."""
        return self.cpu_memory_gb + (self.tmpfs_mem_gb or 0)

    @property
    def all_cpu_memory_mb(self) -> int:
        """All Memory in MB."""
        # while GB is sometimes 1000^3 instead of 1024^3 (typically HD size etc), for memory, base 1024 is always used.
        # (and docker seems to use 1000 based, which is silly)
        return (self.all_cpu_memory_gb or 0) * 1024

    @property
    def all_cpu_memory_byte(self) -> int:
        """All Memory in bytes."""
        # while GB is sometimes 1000^3 instead of 1024^3 (typically HD size etc), for memory, base 1024 is always used.
        # (and docker seems to use 1000 based, which is silly)
        return (self.all_cpu_memory_gb or 0) * 1024 * 1024 * 1024

    def sanitized_copy(self, logged_in: bool = True, _same_project: bool = False) -> "JobStateResources":
        """Anonymized copy."""
        # not entirely public, but all not that private
        sanitazion_updates: dict[str, Any] = {
            "ssh_username": "hidden",
            "ssh_proxy_username": "hidden",
            "port_mappings": [],
            "gpu_details": None,
        }
        if not logged_in:
            sanitazion_updates |= {
                "gpu_ids": [],
                "cpu_ids": [],
                "slave_host": "hidden",
                "slave_name": "hidden",
                "slave_instance_id": "hidden",
                "slave_instance_pid": -1,
                "worker_id": -1,
                "ssh_host": "hidden",
                "ssh_port": -1,
                "ssh_proxy_host": "hidden",
                "ssh_proxy_port": -1,
            }

        return self.model_copy(update=sanitazion_updates)


class MaxSimultaneousJobs(BaseSchema):
    """Maximum number of simultaneous jobs."""

    bucket_name: str = Field(pattern=r"^[\w][\w_\-@.]*$")
    bucket_max: int = Field(..., gt=0)


def _check_ssh_pub_key(v: str) -> str:
    if not is_valid_ssh_key(v):
        msg = "Invalid SSH public key"
        raise ValueError(msg)
    return v


PublicSshKey = Annotated[str, AfterValidator(_check_ssh_pub_key)]


class JobRequestExtra(BaseSchema):
    """Job Request Extras."""

    ssh_pub_keys: list[PublicSshKey] = Field(default_factory=list)
    email_on_queue: list[EmailStr] = Field(default_factory=list)  # also email on queueing of restart job
    email_on_run: list[EmailStr] = Field(default_factory=list)  # no email on restart
    email_on_end: list[EmailStr] = Field(default_factory=list)  # no email on successful halt
    email_on_halt: list[EmailStr] = Field(default_factory=list)  # email on MUSTHALT
    email_on_restart: list[EmailStr] = Field(default_factory=list)

    def sanitized_copy(self, _logged_in: bool = True, _same_project: bool = False) -> "JobRequestExtra":
        """Anonymized copy."""
        # hide all
        return JobRequestExtra(
            ssh_pub_keys=[],
            email_on_queue=[],
            email_on_run=[],
            email_on_end=[],
            email_on_halt=[],
            email_on_restart=[],
        )


MINIMUM_MAX_DURATION_SECONDS = 60


def valid_duration(v: str) -> str:
    """Validate duration."""
    if duration_string_to_seconds(v) is None:
        msg = (
            f"Invalid duration: {v}. Must be a number followed by a time unit "
            "(examples: '1h', '5 minutes', '3 days', etc.)."
        )
        raise ValueError(msg)
    return v


DurationStr = Annotated[str, AfterValidator(valid_duration)]


class JobRequestSchedulingBase(BaseSchema):
    """GPULab scheduling instructions Base."""

    interactive: bool = False
    """Interactive jobs will either run immediately, or fail directly. They will never be QUEUED for a long time."""
    min_duration: Annotated[
        DurationStr | None,
        Field(validation_alias=AliasChoices("minDuration", "killableAfter")),
    ]
    """GPULab might stop this job after this duration.

    Setting this allows GPULab to schedule this job earlier than it can otherwise,
    but there is a chance your job will be stopped after this time.

    (If it is restartable, it will however restart later.)

    Format: a number followed by a unit (ex: 5 minutes, 3 hour, 2 days, 1 week, ...)
    """

    # For a while JobRequestScheduling.min_duration was named JobRequestScheduling.killable_after
    # We want to be 100% backward compatible to this

    restartable: bool = False
    """Restartable jobs can be stopped and later restarted by GPULab. Before stopping,
    GPULab will send a signal to the job.
    In exchange for this flexibility, GPULab can start the jobs sooner than it otherwise would.
    """

    allow_halt_without_signal: bool = False  # If True, the "halting procedure" is not needed for this job
    """Normally, jobs that are restartable are send a signal and given some time to finish.

    If allow_halt_without_signal is set to true, the GPULab can just stop the job without warning,
    and still restart it cleanly. This is used to flag jobs that don't need a clean exit to be restartable."""

    reservation_ids: Annotated[list[ReservationId], EnsureListValidator] = Field(
        default_factory=list,
        validation_alias=AliasChoices("reservationIds", "reservationId"),
    )
    """The reservation ID(s) to use it for the job.
    This allows the job to start when it otherwise would not be able to start due to a reservation."""

    max_duration: DurationStr | None
    """ The maximum duration of this job. GPULab will always stop your job after this time.
    If the job has been restarted once or more, the total duration of all the job runs is used.

    Format: a number followed by a unit (ex: 5 minutes, 3 hour, 2 days, 1 week, ...)"""

    @field_validator("max_duration")
    @classmethod
    def _validate_max_duration(cls, max_duration: str | None) -> str | None:
        if max_duration:
            max_dur_s = duration_string_to_seconds(max_duration)
            assert max_dur_s  # already tested in the previous validator
            if max_dur_s < MINIMUM_MAX_DURATION_SECONDS:
                msg = f'max_duration must be at least 1 minute, not "{max_duration}"'
                raise ValueError(msg)

        return max_duration

    max_simultaneous_jobs: MaxSimultaneousJobs | None = None
    """Control how many of your jobs can run at the same time.

    Format: { "bucket_name": <bucketname>, "bucket_max": <max number> }"""

    not_before: AwareDatetime | None = None
    """Request GPULab not to start the job before a specified date.

    It will stay QUEUED at least until the requested time.

    Format: an RFC3339 date."""

    not_after: AwareDatetime | None = None
    """Request GPULab to FAIL the job if it is still QUEUED at a specified date.

    Does not affect an already running job, only prevents start after the date.)

    Format: an RFC3339 date."""

    @field_validator("not_before", "not_after")
    @classmethod
    def _check_combination_with_interactive(
        cls, v: datetime.datetime | None, info: ValidationInfo
    ) -> datetime.datetime | None:
        if v and info.data["interactive"]:
            msg = f"{info.field_name} cannot be combined with interactive"
            raise ValueError(msg)
        return v

    @field_validator("not_after")
    @classmethod
    def _check_order(cls, not_after: datetime.datetime | None, info: ValidationInfo) -> datetime.datetime | None:
        not_before = info.data.get("not_before")
        if not_after and not_before and not_before >= not_after:
            msg = f"not_before ({not_before}) must be before not_after ({not_after})"
            raise ValueError(msg)
        return not_after

    # # cached_property creates a lot of problems. For example, .model_copy just copies it.
    # # pydantic also doesn't like that it gets deleted on frozen instances,
    # # which is how you can manually invalidate it.
    # @property
    # def max_duration_s(self) -> int | None:
    #     """Maximum duration in seconds."""
    #     return (
    #         duration_string_to_seconds(self.max_duration)
    #         if self.max_duration is not None
    #         else None
    #     )

    # # cached_property creates a lot of problems. For example, .model_copy just copies it.
    # # pydantic also doesn't like that it gets deleted on frozen instances,
    # # which is how you can manually invalidate it.
    # @property
    # def min_duration_s(self) -> int | None:
    #     """Min duration in seconds."""
    #     return (
    #         duration_string_to_seconds(self.min_duration)
    #         if self.min_duration is not None
    #         else None
    #     )


class NewJobRequestScheduling(JobRequestSchedulingBase):
    """GPULab scheduling instructions for new Jobs."""

    min_duration: Annotated[
        DurationStr | None,
        Field(validation_alias=AliasChoices("minDuration", "killableAfter")),
    ] = None
    """GPULab might stop this job after this duration.

    Setting this allows GPULab to schedule this job earlier than it can otherwise,
    but there is a chance your job will be stopped after this time.

    (If it is restartable, it will however restart later.)

    Format: a number followed by a unit (ex: 5 minutes, 3 hour, 2 days, 1 week, ...)
    """

    max_duration: DurationStr | None = None
    """ The maximum duration of this job. GPULab will always stop your job after this time.
    If the job has been restarted once or more, the total duration of all the job runs is used.

    Format: a number followed by a unit (ex: 5 minutes, 3 hour, 2 days, 1 week, ...)"""


class JobRequestScheduling(JobRequestSchedulingBase):
    """GPULab scheduling instructions."""

    min_duration: Annotated[  # pyright: ignore [reportIncompatibleVariableOverride] variable is immutable due to frozen base model, so this is fine
        DurationStr,
        Field(
            validation_alias=AliasChoices("minDuration", "killableAfter"),
        ),
    ]

    """GPULab might stop this job after this duration.

    Setting this allows GPULab to schedule this job earlier than it can otherwise,
    but there is a chance your job will be stopped after this time.

    (If it is restartable, it will however restart later.)

    Format: a number followed by a unit (ex: 5 minutes, 3 hour, 2 days, 1 week, ...)
    """

    # For a while JobRequestScheduling.min_duration was named JobRequestScheduling.killable_after
    # We want to be 100% backward compatible to this

    max_duration: DurationStr  # pyright: ignore [reportIncompatibleVariableOverride] variable is immutable due to frozen base model, so this is fine
    """ The maximum duration of this job. GPULab will always stop your job after this time.
    If the job has been restarted once or more, the total duration of all the job runs is used.

    Format: a number followed by a unit (ex: 5 minutes, 3 hour, 2 days, 1 week, ...)"""

    @field_validator("max_duration")
    @classmethod
    def _validate_max_duration(cls, max_duration: str | None) -> str:
        if max_duration is None:
            msg = "max_duration is mandatory"
            raise ValueError(msg)
        max_dur_s = duration_string_to_seconds(max_duration)
        assert max_dur_s  # already tested in the previous validator
        if max_dur_s < MINIMUM_MAX_DURATION_SECONDS:
            msg = f'max_duration must be at least 1 minute, not "{max_duration}"'
            raise ValueError(msg)
        return max_duration

    # cached_property creates a lot of problems. For example, .model_copy just copies it.
    # pydantic also doesn't like that it gets deleted on frozen instances, which is how you can manually invalidate it.
    @property
    def max_duration_s(self) -> int:
        """Maximum duration in seconds."""
        res = duration_string_to_seconds(self.max_duration)
        if res is None:
            # This should not occur as max_duration is validated
            msg = "max_duration is invalid"
            raise ValueError(msg)
        assert isinstance(res, int)
        return res

    # cached_property creates a lot of problems. For example, .model_copy just copies it.
    # pydantic also doesn't like that it gets deleted on frozen instances, which is how you can manually invalidate it.
    @property
    def min_duration_s(self) -> int:
        """Min duration in seconds."""
        res = duration_string_to_seconds(self.min_duration)
        if res is None:
            # This should not occur as min_duration is validated
            msg = "min_duration is invalid"
            raise ValueError(msg)
        assert isinstance(res, int)
        return res

    def get_min_duration_delta(self, *, default_s: int | None) -> datetime.timedelta:
        """Min duration as timedelta."""
        if self.min_duration is None:
            # Note: self.min_duration is None should never occur!
            if default_s is None:
                msg = "None not allowed for default_s"
                raise ValueError(msg)
            return datetime.timedelta(seconds=default_s)

        res = duration_string_to_seconds(self.min_duration)
        if res is None:
            msg = "min_duration is invalid"
            raise ValueError(msg)
        return datetime.timedelta(seconds=res)

    def sanitized_copy(self, _logged_in: bool = True, _same_project: bool = False) -> "JobRequestScheduling":
        """Anonymized copy."""
        # hide only reservation ID
        return self.model_copy(
            update={
                "reservation_ids": ["00000000-0000-0000-0000-000000000000"] if self.reservation_ids else [],
                "max_simultaneous_jobs": None,
            }
        )


class JobStateScheduling(BaseSchema):
    """Job State Scheduling."""

    assigned_cluster_id: int | None = None
    assigned_instance_id: str | None = None
    assigned_slave_name: str | None = None

    queued_explanations: list[str] = Field(default_factory=list)
    within_max_simultaneous_jobs: Annotated[bool | None, Field(deprecated=True, exclude=True)] = None
    # deprecated now! Scheduling determines this internally on the fly instead

    tally_increment: float | None = None
    """The tally increment each 5 minutes caused by the resource use of this job"""

    halt_events: list[HaltInfo] = Field(default_factory=list)
    """Halt events describe why which job(s) where stopped (for which job(s))
    They can be added when a job is started because other jobs were halted, or when a job is halted.
    """

    # The 3 schedulers look at this job at different times. (basic scheduler, advanced score/prio based halt scheduler)
    # These values might be None, if the corresponding scheduler part hasn't seen the job yet.
    # If this is not none, the corresponding scheduler part has looked at least once.
    # Some values _might_ not be updated after the first time the corresponding scheduler part sees them.
    #    (because that info is not needed, and there's a cost to updating it.)
    scheduler_seen_base: datetime.datetime | None = None
    scheduler_seen_score_halt: datetime.datetime | None = None
    """Updated when scheduler actually checks if job can be started by halting others."""
    scheduler_seen_prio_halt: datetime.datetime | None = None
    """updated when scheduler actually checks if job can be started by halting others"""

    ignoring_reservation_ids: list[str] = Field(default_factory=list)
    """The scheduler is letting this job use these reservations, even though the job is not allowed too use them.
    This makes the job a target to automatically HALT when users of the reservation start jobs.

    (At the moment, the scheduler allows jobs to ignore only 1 reservation at a time.
     This field is future proof by allowing multiple.)
    """

    def sanitized_copy(self, _logged_in: bool = True, _same_project: bool = False) -> "JobStateScheduling":
        """Anonymized copy."""
        # Nothing to hide
        return self.model_copy()


def _allow_single_value(v: Any) -> Any:
    if isinstance(v, str):
        return [v]
    return v


class JobRequestResources(BaseSchema):
    """Job Request Resources."""

    cpus: int = Field(..., gt=0)
    gpus: int = Field(..., ge=0)
    cpu_memory_gb: int = Field(..., gt=0)
    gpu_memory_gb: int | None = Field(
        None,
        gt=1,
    )

    features: Annotated[
        list[Annotated[str, StringConstraints(min_length=2)]],
        BeforeValidator(_allow_single_value),
    ] = Field(default_factory=list)
    gpu_model: Annotated[
        list[Annotated[str, StringConstraints(min_length=2)]],
        BeforeValidator(_allow_single_value),
    ] = Field(default_factory=list)

    min_cuda_version: int | None = Field(None, ge=0)
    cluster_id: int | None = Field(None, ge=0)
    slave_name: Annotated[str, StringConstraints(min_length=1)] | None = None
    slave_instance_id: Annotated[str, StringConstraints(min_length=1)] | None = None

    def sanitized_copy(self, _logged_in: bool = True, _same_project: bool = False) -> "JobRequestResources":
        """Copy with removed confidential data.

        :param logged_in: sanitized copy for logged in user, or for anonymous user?
        :param same_project: sanitized copy for user in same project, or for someone else?
        """
        return self.model_copy()  # nothing to sanitize

    @property
    def cluster_id_list(self) -> list[int]:
        """Cluster ID as list."""
        return [self.cluster_id] if self.cluster_id else []

    @property
    def cpu_memory_mb(self) -> int:
        """CPU Memory in MB."""
        # while GB is sometimes 1000^3 instead of 1024^3 (typically HD size etc), for memory, base 1024 is always used.
        # (and docker seems to use 1000 based, which is silly)
        return self.cpu_memory_gb * 1024

    @property
    def cpu_memory_byte(self) -> int:
        """CPU Memory in bytes."""
        # while GB is sometimes 1000^3 instead of 1024^3 (typically HD size etc), for memory, base 1024 is always used.
        # (and docker seems to use 1000 based, which is silly)
        return self.cpu_memory_gb * 1024 * 1024 * 1024

    def matches_gpu_model(self, tested_model: str) -> bool:
        """Check if a certain GPU is compatible with the ones requested."""
        return any(my_model.lower() in tested_model.lower() for my_model in self.gpu_model)


def _normalize_path(dir: str) -> str:
    return dir + "/" if not dir.endswith("/") else dir


class TmpfsJobStorage(BaseSchema):
    """tmpfs Job Storage."""

    host_path: Literal["tmpfs"]
    container_path: Annotated[str, AfterValidator(_normalize_path)]
    size_gb: int = Field(..., gt=0)

    @property
    def is_project_share_auto(self) -> bool:
        """Check if is magic project share."""
        return False

    @property
    def is_ssh_dir(self) -> bool:
        """Check if is SSH dir."""
        return False


class JobStorage(BaseSchema):
    """Normal directory Job Storage."""

    container_path: Annotated[str, AfterValidator(_normalize_path)] | None = None
    host_path: Annotated[str, AfterValidator(_normalize_path)] | None = None

    # @model_validator(mode="before")
    # @classmethod
    # def check_either_container_path_or_host_path(cls, data: Any) -> Any:
    #     """Make container_path and host_path each others default if only 1 of them is missing."""
    #     if isinstance(data, dict):
    #         if "containerPath" in data and "hostPath" not in data:
    #             data["hostPath"] = data["containerPath"]
    #         if "containerPath" not in data and "hostPath" in data:
    #             data["containerPath"] = data["hostPath"]
    #     return data

    @model_validator(mode="before")
    @classmethod
    def check_either_container_path_or_host_path(cls, data: Any) -> Any:
        """Make sure at least 1 of container_path and host_path is specified."""
        if (
            isinstance(data, dict)
            and "containerPath" not in data
            and "hostPath" not in data
            and "container_path" not in data
            and "host_path" not in data
        ):
            msg = "either containerPath or hostPath must be specified"
            raise ValueError(msg)
        return data

    @model_validator(mode="after")
    def _validate_ssh(self) -> "JobStorage":
        """Support binding of .ssh folder."""
        if self.host_path == ".ssh/" and not self.container_path:
            object.__setattr__(self, "container_path", "/root/.ssh/")
        if self.container_path == "/root/.ssh/" and not self.host_path:
            object.__setattr__(self, "host_path", ".ssh/")

        return self

    @model_validator(mode="after")
    def _check_has_data(self) -> "JobStorage":
        if not self.container_path and not self.host_path:
            msg = "need at least one of hostPath or containerPath"
            raise ValueError(msg)

        if not self.container_path:
            object.__setattr__(self, "container_path", self.host_path)
        if not self.host_path:
            object.__setattr__(self, "host_path", self.container_path)
        return self

    @property
    def is_project_share_auto(self) -> bool:
        """Check if is magic project share."""
        return self.host_path == "PROJECT_SHARE_AUTO"

    @property
    def is_ssh_dir(self) -> bool:
        """Check if is SSH dir."""
        return self.host_path == ".ssh/" or (not self.host_path and self.container_path == "/root/.ssh/")

    @classmethod
    def from_string(cls, s: str) -> "JobStorage":
        """Construct from a string."""
        return cls(container_path=s, host_path=s)


def _determine_job_storage_type(v: Any) -> str:
    if isinstance(v, dict):
        return "tmpfs" if v.get("hostPath") == "tmpfs" else "directory"

    return "tmpfs" if getattr(v, "host_path", None) == "tmpfs" else "directory"


AnyJobStorage = Annotated[
    Annotated[TmpfsJobStorage, Tag("tmpfs")] | Annotated[JobStorage, Tag("directory")],
    Discriminator(_determine_job_storage_type),
]


def _convert_int_portmapping(v: Any) -> Any:
    if isinstance(v, int):
        return {"containerPort": v}
    return v


def _convert_str_storage(v: Any) -> Any:
    if isinstance(v, str):
        return {"hostPath": v}
    return v


class JobRequestDocker(BaseSchema):
    """Job Request: Docker."""

    image: Annotated[str, StringConstraints(pattern=DOCKER_IMAGE_PATTERN)]
    command: str | list[str] = Field(default_factory=list)
    """A single command not in a list has a different meaning than a single command in a list (!):

    In the 1st case, the command can includes spaces and is processed like a shell
    In the 2nd case, the first element is the executable and the other the literal arguments
    Note that dockerpy container.run directly accepts the same 2 types of arguments in the same way
    """

    environment: dict[
        Annotated[str, StringConstraints(pattern=r"^[a-zA-Z_][a-zA-Z0-9_]*$")],
        str | int | bool,
    ] = Field(default_factory=dict)
    storage: list[Annotated[AnyJobStorage, BeforeValidator(_convert_str_storage)]] = Field(default_factory=list)
    port_mappings: list[Annotated[JobPortMapping, BeforeValidator(_convert_int_portmapping)]] = Field(
        default_factory=list
    )
    project_gid_variable_name: Annotated[str, StringConstraints(pattern=r"^[a-zA-Z_][a-zA-Z0-9_]*$")] | None = None
    """Environment variable name in which the project GID should be put."""
    working_dir: Annotated[str, StringConstraints(pattern=r"^/([^\n/]+/)*$")] | None = None
    group_add: Annotated[list[Annotated[str, StringConstraints(pattern=r"^\w+$")]], EnsureListValidator] = Field(
        default_factory=list
    )
    user: Annotated[str, StringConstraints(pattern=r"^\w+$")] | None = None

    @property
    def tmpfs_memory_gb(self) -> int:
        """Total tmpfs memory in GB."""
        return sum(s.size_gb if isinstance(s, TmpfsJobStorage) else 0 for s in self.storage)

    @property
    def tmpfs_memory_mb(self) -> int:
        """Total tmpfs memory in MB."""
        # while GB is sometimes 1000^3 instead of 1024^3 (typically HD size etc), for memory, base 1024 is always used.
        # (and docker seems to use 1000 based, which is silly)
        return self.tmpfs_memory_gb * 1024

    @property
    def tmpfs_memory_byte(self) -> int:
        """Total tmpfs memory in bytes."""
        # while GB is sometimes 1000^3 instead of 1024^3 (typically HD size etc), for memory, base 1024 is always used.
        # (and docker seems to use 1000 based, which is silly)
        return self.tmpfs_memory_gb * 1024 * 1024 * 1024

    @property
    def image_nopass(self) -> str:
        """Docker image with the password removed."""
        match = DOCKER_IMAGE_REGEX.match(self.image)

        if match and match.group("password"):
            return (
                f"{match.group('username')}:XXXX@{match.group('repository')}{match.group('image')}:{match.group('tag')}"
            )

        return self.image

    @property
    def command_as_str(self) -> str | None:
        """Command as a string."""
        if not self.command:
            return None
        if isinstance(self.command, str):
            return self.command
        return " ".join(self.command)

    def sanitized_copy(self, logged_in: bool = True, same_project: bool = False) -> "JobRequestDocker":
        """Anonymized copy."""
        if logged_in and same_project:
            return self.model_copy(
                update={
                    "image": self.image_nopass,  # password is stripped from docker image if shown to project members
                }
            )
        return JobRequestDocker(
            image="hidden",
        )


class JobRequestBase(BaseSchema):
    """Job Request Base."""

    resources: JobRequestResources
    docker: JobRequestDocker
    scheduling: JobRequestSchedulingBase
    extra: JobRequestExtra = Field(default_factory=lambda: JobRequestExtra())

    @property
    def all_cpu_memory_gb(self) -> int:
        """Sum of all CPU Memory in GB."""
        return self.resources.cpu_memory_gb + self.docker.tmpfs_memory_gb

    @property
    def all_cpu_memory_mb(self) -> int:
        """Sum of all CPU Memory in MB."""
        # while GB is sometimes 1000^3 instead of 1024^3 (typically HD size etc), for memory, base 1024 is always used.
        # (and docker seems to use 1000 based, which is silly)
        return self.all_cpu_memory_gb * 1024

    @property
    def all_cpu_memory_byte(self) -> int:
        """Sum of all CPU Memory in bytes."""
        # while GB is sometimes 1000^3 instead of 1024^3 (typically HD size etc), for memory, base 1024 is always used.
        # (and docker seems to use 1000 based, which is silly)
        return self.all_cpu_memory_gb * 1024 * 1024 * 1024


class JobRequest(JobRequestBase):
    """Job Request."""

    scheduling: JobRequestScheduling  # pyright: ignore [reportIncompatibleVariableOverride] variable is immutable due to frozen base model, so this is fine

    def sanitized_copy(self, logged_in: bool = True, same_project: bool = False) -> "JobRequest":
        """Copy with removed confidential data.

        :param logged_in: sanitized copy for logged in user, or for anonymous user?
        :param same_project: sanitized copy for user in same project, or for someone else?
        """
        return JobRequest(
            resources=self.resources.sanitized_copy(logged_in, same_project),
            docker=self.docker.sanitized_copy(logged_in, same_project),
            scheduling=self.scheduling.sanitized_copy(logged_in, same_project),
            extra=self.extra.sanitized_copy(logged_in, same_project),
        )


class NewJobRequest(JobRequestBase):
    """Job Request for new jobs."""

    scheduling: NewJobRequestScheduling = Field(  # pyright: ignore [reportIncompatibleVariableOverride] variable is immutable due to frozen base model, so this is fine
        default_factory=lambda: NewJobRequestScheduling()
    )


class RestartInfo(BaseSchema):
    """Restart Information.

    Available in jobs that have been restarted after being halted.
    """

    initial_job_uuid: JobId
    restart_count: int = Field(..., ge=0)


IdlabUserType = Literal[
    "stud-idlab-ghent",
    "stud-idlab-antwerp",
    "idlab-ghent",
    "idlab-antwerp",
    "non-idlab",
]


class UserDetails(BaseSchema):
    """User Details."""

    first_name: str | None = None
    last_name: str | None = None
    portal_home: str | None = None
    organization: str | None = None
    affiliation: str | None = None
    # UserDetails is considered legacy (JobOwnerV5 does not have it):
    #    so we'll grudgingly allow the non-aware datetime that are returned by the legacy code
    # creation_date: AwareDatetime | None = None
    creation_date: datetime.datetime | None = None
    eppn: str | None = None
    country: str | None = None
    idlab: IdlabUserType | None = None
    student: bool | None = None


USER_URN_PATTERN = r"urn:publicid:IDN\+(?P<authority>[^+:]+)(:(?P<subauthority>[^+]+))?\+user\+(?P<subject>[^+]+)$"
PROJECT_URN_PATTERN = (
    r"urn:publicid:IDN\+(?P<authority>[^+:]+)(:(?P<subauthority>[^+]+))?\+project\+(?P<subject>[^+]+)$"
)


class JobOwnerBase(BaseSchema):
    """Job Owner base fields, shared between NewJobOwner and JobOwner."""

    project_urn: str = Field(pattern=PROJECT_URN_PATTERN)

    @property
    def project_name(self) -> str:
        """Name from Project URN."""
        return URN(urn=self.project_urn).name  # type: ignore[no-any-return]


class NewJobOwnerV4(JobOwnerBase):
    """Job Owner specified in new job request."""

    # Only for backward compatibility with v4
    user_urn: str | None = Field(None, deprecated=True, exclude=True)
    user_email: str | None = Field(None, deprecated=True, exclude=True)
    user_details: Any | None = Field(None, deprecated=True, exclude=True)


class NewJobOwnerV5(JobOwnerBase):
    """Job Owner specified in new job request."""


class JobOwner(JobOwnerBase):
    """Job Owner."""

    user_urn: str = Field(pattern=USER_URN_PATTERN)
    user_email: EmailStr
    user_details: UserDetails | None = None

    # For forward compatibility with JobOwnerV5
    user_id: str | None = None
    project_id: str | None = None
    experiment_id: str | None = None

    @property
    def userurn_auth(self) -> str:
        """Authority from User URN."""
        res = urn_to_auth(self.user_urn)
        if not res:
            msg = "Invalid User URN"
            raise ValueError(msg)
        return res

    @property
    def userurn_name(self) -> str:
        """Name from User URN."""
        res = urn_to_name(self.user_urn)
        if not res:
            msg = "Invalid User URN"
            raise ValueError(msg)
        return res

    @property
    def user_mini_id(self) -> str:
        """Mini-ID computed from User URN."""
        return urn_to_user_mini_id(self.user_urn) or "error"

    @property
    def project_name(self) -> str:
        """Name from Project URN."""
        return URN(urn=self.project_urn).name  # type: ignore[no-any-return]

    def is_partially_sanitized(self) -> bool:
        """Check if partially sanitized."""
        return (
            self.project_urn == "urn:publicid:IDN+hidden+project+hidden"
            or self.user_urn == "urn:publicid:IDN+hidden+user+hidden"
            or self.user_email == "hidden@hidden.hidden"
        )

    def sanitized_copy(self, logged_in: bool = True, same_project: bool = False) -> "JobOwner":
        """Anonymized copy."""
        # hide the user details, email address and project
        if logged_in and same_project:
            return self.model_copy(
                update={
                    "user_urn": self.user_urn,
                    "user_email": "hidden@hidden.hidden",
                    "project_urn": self.project_urn,
                    "user_details": None,
                }
            )
        return self.model_copy(
            update={
                "user_urn": "urn:publicid:IDN+hidden+user+hidden",
                "user_email": "hidden@hidden.hidden",
                "project_urn": "urn:publicid:IDN+hidden+project+hidden",
                "user_details": None,
                "user_id": None,
                "project_id": None,
                "experiment_id": None,
            }
        )


class JobOwnerV5(BaseSchema):
    """Job Owner. Slices version."""

    user_id: str
    project_id: str
    # experiment_id = None will temporarily be supported for DB backward compatibility in DB.
    experiment_id: str | None = None


# State vs Status
#
#   -> in the english language: very related, and often synonyms
#   -> typically in technical language, state is used in a more broad sense, and status is more "one-dimensional"
#       -> thus you can have a state containing multiple statuses
#
#  Here:
#   -> in JobState we describe the entire variable state of the Job.
#      "status" holds the ID of the current discrete step in the Job's lifecycle ("the workflow of job execution")
#      so "lifecycle_step" or "workflow_position" would be a synonym for our "status", but both feels too convoluted
#
class JobState(BaseSchema):
    """Current Job State."""

    status: JobStatus  # The ID of the current step in the Job's lifecycle
    scheduling: JobStateScheduling  # mandatory, but content can all be None
    event_times: JobEventTimes  # mandatory, but content can be empty
    resources: JobStateResources | None = None  # only filled in once job is at least STARTING
    final_usage_statistics: GPULabUsageStatistics | None = None
    waste_review: WasteReview | None = None  # Waste report calculated on master server using clickhouse

    waste_report: Annotated[Any, Field(None, deprecated=True, exclude=True)] = None
    """wasteReport should be ignored and removed.

    For backward compatibility only."""

    # updatable fields: FIELDNAME_PORT_MAPPINGS, FIELDNAME_GPU_INFO, FIELDNAME_END_DATE, FIELDNAME_SUMMARY_STATISTICS

    def sanitized_copy(self, logged_in: bool = True, same_project: bool = False) -> "JobState":
        """Anonymized copy."""
        # most is public here
        return JobState(
            status=self.status,
            resources=self.resources.sanitized_copy(logged_in, same_project) if self.resources else None,
            scheduling=self.scheduling.sanitized_copy(logged_in, same_project),
            event_times=self.event_times.sanitized_copy(logged_in, same_project),
            final_usage_statistics=self.final_usage_statistics.model_copy()
            if self.final_usage_statistics and logged_in
            else None,
            waste_review=self.waste_review,
        )


JBShape = TypeVar("JBShape", bound="JobBase")


class JobBase(BaseSchema):
    """GPULab Job."""

    id: Annotated[JobId, Field(validation_alias="uuid", serialization_alias="uuid")]

    # cached_property creates a lot of problems. For example, .model_copy just copies it.
    # pydantic also doesn't like that it gets deleted on frozen instances, which is how you can manually invalidate it.
    # @cached_property
    @property
    def uuid(self) -> UUID:
        """Job ID as UUID."""
        return UUID(self.id)

    name: str = Field(..., min_length=1)
    description: str | None = None
    deployment_environment: str | None = Field("production", deprecated=True)
    request: JobRequest

    state: JobState
    restart_info: RestartInfo | None = None

    @model_validator(mode="after")
    def _init_restart_info(self: JBShape) -> JBShape:
        # restart_info is known explicitly for jobs with uuid that do not have it.
        # to make things auto-consistent, we add it here automatically.

        if self.id and not self.restart_info:
            object.__setattr__(
                self,
                "restart_info",
                RestartInfo(initial_job_uuid=self.id, restart_count=0),
            )

        return self

    # backward compatible check if job is "stable" aka "production"
    @property
    @deprecated("deployment_environment is no longer used.")
    def is_production(self) -> bool:
        """Check if production.

        For backward compatibility only.
        """
        return self.deployment_environment in ["stable", "prod", "production"]

    def replace_event_times_attrs(self: JBShape, **kwargs: Any) -> JBShape:
        """Update event times in Job."""
        return self.model_copy(
            update={
                "state": self.state.model_copy(
                    update={
                        "event_times": self.state.event_times.model_copy(update=kwargs),
                    }
                )
            }
        )

    def replace_request_resources_attrs(self: JBShape, **kwargs: Any) -> JBShape:
        """Update request resources in job."""
        return self.model_copy(
            update={
                "request": self.request.model_copy(
                    update={
                        "resources": self.request.resources.model_copy(update=kwargs),
                    }
                )
            }
        )

    def replace_request_scheduling_attrs(self: JBShape, **kwargs: Any) -> JBShape:
        """Update request scheduling in job."""
        return self.model_copy(
            update={
                "request": self.request.model_copy(
                    update={
                        "scheduling": self.request.scheduling.model_copy(update=kwargs),
                    }
                )
            }
        )

    def replace_request_extra_attrs(self: JBShape, **kwargs: Any) -> JBShape:
        """Update request extra in job."""
        return self.model_copy(
            update={
                "request": self.request.model_copy(
                    update={
                        "extra": self.request.extra.model_copy(update=kwargs),
                    }
                )
            }
        )

    def replace_state_scheduling(self: JBShape, new_scheduling: JobStateScheduling) -> JBShape:
        """Replace state scheduling."""
        return self.model_copy(
            update={
                "state": self.state.model_copy(update={"scheduling": new_scheduling}),
            }
        )

    def replace_state_scheduling_attrs(self: JBShape, **kwargs: Any) -> JBShape:
        """Replace state scheduling attributes."""
        return self.model_copy(
            update={
                "state": self.state.model_copy(
                    update={
                        "scheduling": self.state.scheduling.model_copy(update=kwargs),
                    }
                ),
            }
        )

    def add_halt_info(self: JBShape, halt_info: HaltInfo) -> JBShape:
        """Add halt info."""
        return self.replace_state_scheduling_attrs(halt_events=[*self.state.scheduling.halt_events, halt_info])

    def replace_state_final_usage_statistics(self: JBShape, final_usage_statistics: GPULabUsageStatistics) -> JBShape:
        """Replace finale usage statistics."""
        return self.model_copy(
            update={
                "state": self.state.model_copy(update={"final_usage_statistics": final_usage_statistics}),
            }
        )

    def replace_state_waste_review(self: JBShape, waste_review: WasteReview | None) -> JBShape:
        """Replace state waste review."""
        return self.model_copy(
            update={
                "state": self.state.model_copy(update={"waste_review": waste_review}),
            }
        )

    def replace_state_resources(self: JBShape, new_resources: JobStateResources) -> JBShape:
        """Replace state resources."""
        return self.model_copy(
            update={
                "state": self.state.model_copy(update={"resources": new_resources}),
            }
        )

    def replace_state_resources_fields(self: JBShape, **kwargs: Any) -> JBShape:
        """Replace state resource fields."""

        assert self.state.resources
        return self.model_copy(
            update={
                "state": self.state.model_copy(
                    update={
                        "resources": self.state.resources.model_copy(update=kwargs),
                    }
                ),
            }
        )

    @property
    def any_cluster_id(self) -> int | None:
        """Retrieve any available cluster id from this job."""

        # Effective cluster id takes precedence
        if self.state:
            if self.state.resources and self.state.resources.cluster_id:
                return self.state.resources.cluster_id
            if self.state.scheduling and self.state.scheduling.assigned_cluster_id:
                return self.state.scheduling.assigned_cluster_id
        return self.request.resources.cluster_id

    @property
    def short_uuid(self) -> str | None:
        """Short ID that is derived from full uuid."""
        if not self.id:
            return None
        if "-" in self.id:
            return self.id[: self.id.index("-")]
        return self.id


def _censor_job_name(name: str, id: str | None, *, same_project: bool = False) -> str:
    return name if same_project or name == "JupyterHub-singleuser" else ("job-" + id[:6] if id else "job")


class Job(JobBase):
    """GPULab Job (legacy)."""

    owner: JobOwner
    """Job Owner.

    Note: Owner is mandatory, except for the Job specified by the user.
          The client will add the owner info based on the PEM and specified project,
          before sending the Job to GPULab master.
          GPULab will check the JobOwner user against the authorized URN"""

    def replace_owner_attrs(self, **kwargs: Any) -> "Job":
        """Update owner in job."""
        return self.model_copy(
            update={"owner": self.owner.model_copy(update=kwargs) if self.owner else JobOwner(**kwargs)}
        )

    def sanitized_copy(self, logged_in: bool = True, same_project: bool = False) -> "Job":
        """Copy with removed confidential data.

        :param logged_in: sanitized copy for logged in user, or for anonymous user?
        :param same_project: sanitized copy for user in same project, or for someone else?
        """
        return self.__class__(
            id=self.id,
            name=_censor_job_name(self.name, self.id, same_project=same_project),
            deployment_environment=self.deployment_environment,
            request=self.request.sanitized_copy(logged_in, same_project),
            description=self.description if logged_in and same_project else None,
            owner=self.owner.sanitized_copy(logged_in, same_project),
            state=self.state.sanitized_copy(logged_in, same_project),
            restart_info=self.restart_info,
        )

    def is_fully_sanitized(self) -> bool:
        """Check if fully sanitized."""
        return bool(self == self.sanitized_copy())

    def is_partially_sanitized(self) -> bool:
        """Check if this job is at least partially sanitized."""
        return (
            self.is_fully_sanitized()
            or self.owner is None  # pyright: ignore [reportAttributeAccessIssue]
            or self.owner.is_partially_sanitized()  # pyright: ignore [reportAttributeAccessIssue]
            or self.request.docker.image == "hidden"
        )


class JobV5(JobBase):
    """GPULab Job (V5 = Slices version 2025).

    Internal version, that always has an owner.
    """

    owner: JobOwnerV5
    """Job Owner."""

    def sanitized_copy(self, logged_in: bool = True, same_project: bool = False) -> "JobV5External":
        """Copy with removed confidential data.

        :param logged_in: sanitized copy for logged in user, or for anonymous user?
        :param same_project: sanitized copy for user in same project, or for someone else?
        """
        return JobV5External(
            id=self.id,
            name=_censor_job_name(self.name, self.id, same_project=same_project),
            deployment_environment=self.deployment_environment,
            request=self.request.sanitized_copy(logged_in, same_project),
            description=self.description if logged_in and same_project else None,
            owner=None,
            state=self.state.sanitized_copy(logged_in, same_project),
            restart_info=self.restart_info,
        )

    def is_fully_sanitized(self) -> bool:
        """Check if fully sanitized."""
        return False

    def is_partially_sanitized(self) -> bool:
        """Check if this job is at least partially sanitized."""
        return self.request.docker.image == "hidden"


class JobV5External(JobBase):
    """GPULab Job (V5 = Slices version 2025).

    Version exposed on endpoints, that might not have an owner.
    """

    owner: JobOwnerV5 | None
    """Job Owner.

    May be None if the owner info is censored for privacy reasons.
    """

    def is_fully_sanitized(self) -> bool:
        """Check if fully sanitized."""
        # TODO we could do more checks, but this is good enough
        return (
            self.owner is None
            and self.request.sanitized_copy() == self.request
            and self.state.sanitized_copy() == self.state
            and self.description is None
        )

    def is_partially_sanitized(self) -> bool:
        """Check if this job is at least partially sanitized."""
        return (
            self.owner is None
            or self.name == _censor_job_name(self.name, self.id)
            or self.request.sanitized_copy() == self.request
            or self.state.sanitized_copy() == self.state
            or self.description is None
        )


class NewJobBase(BaseSchema):
    """New job: schema limited to the values that must be present during initial submission."""

    name: str = Field(..., min_length=1)
    request: NewJobRequest
    description: str | None = None
    state: JobState | None = None
    restart_info: RestartInfo | None = None


class NewJobV4(NewJobBase):
    """New job: schema limited to the values that must be present during initial submission.

    V4 has some backward compatibility (it has fields that are allowed but ignored.).
    """

    owner: NewJobOwnerV4

    # Only for backward compatibility with v4
    deployment_environment: str | None = Field(None, deprecated=True)

    def replace_request_extra_attrs(self, **kwargs: Any) -> "NewJobV4":
        """Update request extra in job."""
        return self.model_copy(
            update={
                "request": self.request.model_copy(
                    update={
                        "extra": self.request.extra.model_copy(update=kwargs),
                    }
                )
            }
        )


class NewJobV4CLI(NewJobBase):
    """New job: schema limited to the values that must be present during initial submission IN THE CLI.

    V4 has some backward compatibility (it has fields that are allowed but ignored.).
    """

    owner: NewJobOwnerV4 | None = None

    # Only for backward compatibility with v4
    deployment_environment: str | None = Field(None, deprecated=True)


class NewJobV5(NewJobBase):
    """New job: schema limited to the values that must be present during initial submission."""

    # V5 must use OAuth with project, so no owner info is required in new job request
    # owner: NewJobOwnerV5
