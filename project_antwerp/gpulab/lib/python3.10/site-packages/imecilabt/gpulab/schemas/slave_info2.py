"""Slave information v2.

This also adds ClusterInfo
It also converts to and from the old SlaveInfo, and can detect if JSO is the old or new version
Improvements:
   - Better naming
   - More consist
   - Always camelCase in JSON
   - Added ClusterInfo and ResourceInfo
   - Forced immutable ("frozen")
   - Types enforced
   - More strict: None (almost) nowhere allowed
   - Uses python dataclasses (python 3.7+) + dataclasses_json library
"""

import datetime
import re
from enum import Enum
from typing import Annotated, Any
from uuid import UUID

from imecilabt_utils.urn_util import URN
from pydantic import (
    AfterValidator,
    AliasChoices,
    BeforeValidator,
    Field,
    ValidationInfo,
    model_validator,
)
from pytz import UTC

from imecilabt.gpulab.schemas.active_jobs_counts import ActiveJobsCounts
from imecilabt.gpulab.schemas.base import AwareDatetime, BaseSchema


class GpuModel(BaseSchema):
    """GPU Model."""

    vendor: str
    name: str
    memory_mb: int

    def __hash__(self) -> int:
        """Hash for making GPUModel work in lists."""
        return hash((self.vendor, self.name, self.memory_mb))


class ResourceInfo(BaseSchema):
    """Slave resource Information."""

    system_total: int = Field(strict=True)
    acquired: int = Field(strict=True)
    used: int = Field(strict=True)
    available: int = Field(strict=True)

    @model_validator(mode="after")
    def _check_count(self) -> "ResourceInfo":
        if self.acquired - self.used != self.available:
            msg = f"acquired - used != available ({self.acquired} - {self.used} != {self.available})"
            raise ValueError(msg)

        return self

    def add(self, other: "ResourceInfo") -> "ResourceInfo":
        """Merge resources from different slaves."""
        return ResourceInfo(
            system_total=self.system_total + other.system_total,
            acquired=self.acquired + other.acquired,
            used=self.used + other.used,
            available=self.available + other.available,
        )

    def merge_instances(self, other: "ResourceInfo") -> "ResourceInfo":
        """Merge resources from different instance on the same slave."""
        assert self.system_total == other.system_total
        return ResourceInfo(
            system_total=self.system_total,
            acquired=self.acquired + other.acquired,
            used=self.used + other.used,
            available=self.available + other.available,
        )

    @classmethod
    def zero(cls) -> "ResourceInfo":
        """Return instance with everything set to 0."""
        return ResourceInfo(
            system_total=0,
            acquired=0,
            used=0,
            available=0,
        )


class SlaveDogAlarmCause(Enum):
    """Slave Watchdog Alarm Cause."""

    RESOURCE_DISAPPEARED = "Resource(s) disappeared"
    JOB_REJECTED = "Job rejected by slave"
    JOB_REQUEST_FAILED = "Failure when slave requested job"
    SLAVE_LOOP_ERROR = "Error in slave loop"
    PERIODIC_REPORT_ERROR = "Error in periodic reporting"
    NET_TIMEOUT = "Network call timeout"
    WATCHDOG_TIMEOUT_MINOR = "Watchdog minor timeout (thread is slow)"
    WATCHDOG_TIMEOUT_MAJOR = "Watchdog major timeout (thread is probably dead)"
    ALIVE_CHECK_FAILED = "Watchdog alive check failed"
    ALIVE_CHECK_RECOVERED = "Watchdog alive check recovered"
    WATCHDOG_MISMATCH = "stop reported before start, or start after start"
    WATCHDOG_INTERNAL = "Internal error in watchdog"


class WatchdogStatistic(BaseSchema):
    """Watchdog Statistic."""

    alive_now: bool = False
    last_alive_date: datetime.datetime | None = None
    alarm_counts: dict[str, int] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("alarmCounts", "alarm_counts"),
    )
    alarm_last_dates: dict[str, AwareDatetime] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("alarmLastDates", "alarm_last_dates"),
    )
    emails_sent: int = 0

    def inc_alarm(self, cause: SlaveDogAlarmCause, now: datetime.datetime | None = None) -> int:
        """Increase the alarm."""
        new_count = self.alarm_counts.get(cause.name, 0) + 1
        self.alarm_counts[cause.name] = new_count
        self.alarm_last_dates[cause.name] = now or datetime.datetime.now(tz=UTC)
        return new_count


class UploadThreadStatistic(BaseSchema):
    """Upload Thread Statistic."""

    cur_len: int
    cur_wait_s: float
    max_ever_len: int
    max_ever_wait_s: float
    total: int
    dropped: int
    retries: int

    def __str__(self) -> str:
        """As string."""
        return (
            f"len={self.cur_len} wait={self.cur_wait_s}s "
            f"(max ever seen: len={self.max_ever_len} wait={self.max_ever_wait_s}s) "
            f"total={self.total} dropped={self.dropped} retries={self.retries}\n"
        )


class SlaveStatistics(BaseSchema):
    """Slave Statistics."""

    upload_threads: dict[str, UploadThreadStatistic]
    watchdog: WatchdogStatistic


class SlaveInstanceBase(BaseSchema):
    """Slave Instance Base."""

    name: str
    instance_id: str
    cluster_id: int  # was allowed to be None, but not anymore
    pid: int | None = Field(None, deprecated=True)
    deployment_environment: str | None = Field(None, deprecated=True)
    software_version: str | None = Field(None, description="Version of the gpulab slave software")

    aliases: list[str] = Field(default_factory=list)
    comment: str | None = None
    host: str | None = None

    @property
    def names(self) -> list[str]:
        "All names."
        return [self.name, *self.aliases]

    def match_name(self, any_name: str | None) -> bool:
        """Check if it does match a name of this slave."""
        if not any_name:
            return False

        return any(any_name.lower() == name.lower() for name in (self.name, self.host, *self.aliases) if name)


STORAGE_PATH_LIMITED_CLUSTER_PROJECTS = "PRIVATE_CLUSTER_PROJECTS"


def _extract_major_cuda_version(full_version: str | None) -> int | None:
    if not full_version:
        return None
    version_pattern = re.compile(r"^([0-9]+)\.[0-9.]+$")
    match_version = version_pattern.match(full_version)
    if match_version:
        return int(match_version.group(1))
    return None


def _retrieve_from_cuda_version_full(cuda_version_major: Any, info: ValidationInfo) -> int | None:
    if isinstance(cuda_version_major, int):
        return cuda_version_major

    return _extract_major_cuda_version(info.data.get("cuda_version_full"))


def _matches_cuda_version_full(cuda_version_major: int | None, info: ValidationInfo) -> int | None:
    cuda_version_full = info.data.get("cuda_version_full")
    if cuda_version_major and cuda_version_full:
        assert cuda_version_major == _extract_major_cuda_version(cuda_version_full), (
            "CUDA major version and full version mismatch."
        )

    return cuda_version_major


class SlaveInfo2(SlaveInstanceBase):
    """Slave Info 2."""

    gpu_model: list[GpuModel]
    cpu_model: list[str]
    worker: ResourceInfo  # system_total is to be ignored here. It will be set the same as acquired.
    cpu_memory_mb: ResourceInfo
    # much more scheduling info will be needed for scheduler to use partial GPUs and their memory.
    # this will need to be added alter
    #   gpu_memory_mb_by_id: List[ResourceInfo]
    gpu: ResourceInfo
    cpu: ResourceInfo
    last_update: AwareDatetime
    shutting_down: bool

    cuda_version_full: str | None = None
    cuda_version_major: Annotated[
        int | None,
        BeforeValidator(_retrieve_from_cuda_version_full),
        AfterValidator(_matches_cuda_version_full),
    ] = Field(None, validate_default=True)

    docker_disk_used_percent: float = -1.0
    accepting_jobs: bool = True
    statistics: SlaveStatistics | None = None
    active_jobs_counts: ActiveJobsCounts | None = None

    # A list of storage paths that are available on the cluster. (optional, where None means "unknown")
    storage_paths_available: list[str] | None = None
    # A dict of storage aliases available on the cluster. (optional, where None means "unknown")
    storage_aliases_available: dict[str, str] | None = None

    # A list of storage paths that are available on the cluster, but only for certain projects.
    # Projects are identified by URN. The special value "PRIVATE_CLUSTER_PROJECTS" is allowed instead of a URN.
    # It means: all projects that have access to the private cluster.
    # Each storage paths may only appear once. Either here as a key, or in storage_paths_available.
    storage_paths_available_project_limited: dict[str, list[str]] | None = None

    # For all the ResourceInfo (cpu_memory_mb, gpu and cpu),
    # the scheduler needs to know which jobs they include, and which not.
    #   empty list -> no jobs (so all resources should be free)
    #   None -> not reported
    active_job_uuids: list[str] | None = None

    def matches_name(self, name: str) -> bool:
        """Check if the given name matches this SlaveInfo2's name or aliases.

        This also ignores all non-alphanum chars!
        """

        def normalize(v: str) -> str:
            return re.sub("[^a-z0-9]", "", v.lower())

        name = normalize(name)
        return any(name == normalize(n) for n in (self.name, *self.aliases))

    def has_storage(self, storage_path: str) -> bool:
        """Check if storage_path is available, including as an alias.

        This does not work correctly with tmpfs or .ssh!
        """

        def add_end_slash_ifneeded(dir: str) -> str:
            return dir + "/" if not dir.endswith("/") else dir

        storage_path = add_end_slash_ifneeded(storage_path)

        if self.storage_paths_available and any(
            storage_path.startswith(add_end_slash_ifneeded(sp)) for sp in self.storage_paths_available
        ):
            return True

        return bool(
            self.storage_aliases_available
            and any(storage_path.startswith(add_end_slash_ifneeded(sp)) for sp in self.storage_aliases_available)
        )

    def allows_storage(  # noqa: C901, PLR0912
        self,
        storage_path: str,
        project_urn: str,
        *,
        cluster_projects_allowed: list[str] | None = None,  # pyright: ignore [reportArgumentType]
    ) -> bool:
        """Check both storage_paths_available and storage_aliases_available.

        This does not work correctly with tmpfs or .ssh!
        :param storage_path:
        :param project_urn: the projects that wants to access the storage
        :param cluster_projects_allowed: the projects allowed in this cluster (or empty list if all)
        :return: whether the storage path is available on this slave, for the specified project.
        """
        if cluster_projects_allowed is None:
            cluster_projects_allowed = []

        def add_end_slash_ifneeded(dir: str) -> str:
            return dir + "/" if not dir.endswith("/") else dir

        storage_path = add_end_slash_ifneeded(storage_path)

        to_check: list[str] = []
        if self.storage_paths_available:
            to_check.extend(
                available
                for available in self.storage_paths_available
                if storage_path.startswith(add_end_slash_ifneeded(available))
            )

        if self.storage_aliases_available:
            for available in self.storage_aliases_available:
                if storage_path.startswith(add_end_slash_ifneeded(available)):
                    to_check.extend((available, self.storage_aliases_available[available]))

        if not to_check:
            return False

        # normalize
        to_check = [add_end_slash_ifneeded(s) for s in to_check]

        if self.storage_paths_available_project_limited:
            for s in to_check:
                for (
                    available,
                    projects,
                ) in self.storage_paths_available_project_limited.items():
                    if s.startswith(add_end_slash_ifneeded(available)):
                        for p in projects:
                            if p == STORAGE_PATH_LIMITED_CLUSTER_PROJECTS:
                                if cluster_projects_allowed:
                                    if URN(project_urn) in [URN(pp) for pp in cluster_projects_allowed]:
                                        return True
                                else:
                                    return True  # all projects allowed on this cluster
                            elif URN(project_urn) == URN(p):
                                return True
                        return False
        return True

    def make_copy_with_added_usage(
        self,
        cpu_memory_mb: int,
        gpu: int,
        cpu: int,
        job_uuids: list[str | UUID] | None = None,
        remove: bool = False,
    ) -> "SlaveInfo2":
        """Make a copy of the job, but modify the gpu, cpu and memory usage by adding to it.

        :param cpu_memory_mb:
        :param gpu:
        :param cpu:
        :param job_uuids:
        :param remove: remove the usage instead of adding it
        :return:
        """
        job_uuids_str: list[str] = list(map(str, job_uuids)) if job_uuids else []

        if remove:
            cpu_memory_mb = -cpu_memory_mb
            cpu = -cpu
            gpu = -gpu

        if self.active_job_uuids is None:
            active_job_uuids = None
        elif remove:
            active_job_uuids = list(set(self.active_job_uuids) - set(job_uuids_str))
        else:
            # add
            active_job_uuids = list(set(self.active_job_uuids + job_uuids_str))

        return self.model_copy(
            update={
                "cpu_memory_mb": ResourceInfo(
                    system_total=self.cpu_memory_mb.system_total,
                    acquired=self.cpu_memory_mb.acquired,
                    used=self.cpu_memory_mb.used + cpu_memory_mb,
                    available=self.cpu_memory_mb.available - cpu_memory_mb,
                ),
                "gpu": ResourceInfo(
                    system_total=self.gpu.system_total,
                    acquired=self.gpu.acquired,
                    used=self.gpu.used + gpu,
                    available=self.gpu.available - gpu,
                ),
                "cpu": ResourceInfo(
                    system_total=self.cpu.system_total,
                    acquired=self.cpu.acquired,
                    used=self.cpu.used + cpu,
                    available=self.cpu.available - cpu,
                ),
                "active_job_uuids": active_job_uuids,
            }
        )

    def make_copy_with_alt_last_update(self, new_last_update: datetime.datetime) -> "SlaveInfo2":
        """Make a copy of the job, but modify last_update."""
        assert new_last_update.tzinfo is not None

        return self.model_copy(update={"last_update": new_last_update})

    def make_copy_with_active_jobs_counts(self, active_jobs_counts: ActiveJobsCounts) -> "SlaveInfo2":
        """Make a copy of the job, but modify active_jobs_counts."""
        return self.model_copy(update={"active_jobs_counts": active_jobs_counts})

    def merge_instances(self, other: "SlaveInfo2") -> "SlaveInfo2":
        """Make new SlaveInfo2 that is a merger of both SlaveInfo's.

        This should only be used to merge info from multiple instances of the same slave

        This is probably only useful if you want to have the resource totals
        """
        assert other.deployment_environment == self.deployment_environment
        assert other.cluster_id == self.cluster_id
        assert other.name == self.name

        preferred = self if self.accepting_jobs and not self.shutting_down else other

        return SlaveInfo2(
            deployment_environment=self.deployment_environment,
            name=self.name,
            aliases=preferred.aliases,
            host=preferred.host,
            instance_id=preferred.instance_id,
            pid=preferred.pid,
            cluster_id=self.cluster_id,
            software_version=preferred.software_version,
            gpu_model=preferred.gpu_model,
            cpu_model=preferred.cpu_model,
            worker=self.worker.merge_instances(other.worker),
            cpu_memory_mb=self.cpu_memory_mb.merge_instances(other.cpu_memory_mb),
            gpu=self.gpu.merge_instances(other.gpu),
            cpu=self.cpu.merge_instances(other.cpu),
            cuda_version_full=preferred.cuda_version_full,
            cuda_version_major=preferred.cuda_version_major,
            last_update=max(self.last_update, other.last_update),
            comment=preferred.comment,
            shutting_down=preferred.shutting_down,
            docker_disk_used_percent=preferred.docker_disk_used_percent,
            accepting_jobs=preferred.accepting_jobs,
            statistics=preferred.statistics,
            storage_paths_available=preferred.storage_paths_available,
            storage_aliases_available=preferred.storage_aliases_available,
            storage_paths_available_project_limited=self.storage_paths_available_project_limited,
            active_job_uuids=list(
                {
                    *(self.active_job_uuids or []),
                    *(other.active_job_uuids or []),
                }
            ),
        )

    def sanitized_copy(self, logged_in: bool = True) -> "SlaveInfo2":
        """Remove statistics for regular users."""

        sanitization_updates: dict[str, Any] = {
            "statistics": None,
        }
        if not logged_in:
            sanitization_updates |= {"active_job_uuids": []}

        return self.model_copy(update=sanitization_updates)


class ClusterInfo(BaseSchema):
    """Cluster Information."""

    deployment_environment: str = Field(..., deprecated=True)
    cluster_id: int
    comment: str
    is_private: bool
    slave_count: int = 0
    gpu_model: list[GpuModel] = Field(default_factory=list)
    worker: ResourceInfo = ResourceInfo.zero()
    cpu_memory_mb: ResourceInfo = ResourceInfo.zero()
    gpu: ResourceInfo = ResourceInfo.zero()
    cpu: ResourceInfo = ResourceInfo.zero()
    active_jobs_counts: ActiveJobsCounts | None = None

    # Does the user requesting the info have access? (optional, None when not relevant)
    have_access: bool | None = None

    # A list of projects that have access (optional, None means public access)
    allowed_projects: list[str] | None = None

    # A list of storage paths that are available on the cluster. (optional, where None means "unknown")
    storage_paths_available: list[str] | None = None
    # A dict of storage aliases available on the cluster. (optional, where None means "unknown")
    storage_aliases_available: dict[str, str] | None = None

    @staticmethod
    def _merge_storage_paths(
        storage_paths_available_1: list[str] | None,
        storage_paths_available_2: list[str] | None,
    ) -> list[str] | None:
        if storage_paths_available_1 is None and storage_paths_available_2 is None:
            return None
        if storage_paths_available_1 is None or storage_paths_available_2 is None:
            return storage_paths_available_1 or storage_paths_available_2

        return list({*storage_paths_available_1, *storage_paths_available_2})

    @staticmethod
    def _merge_storage_aliases(
        storage_aliases_available_1: dict[str, str] | None,
        storage_aliases_available_2: dict[str, str] | None,
    ) -> dict[str, str] | None:
        if storage_aliases_available_1 is None and storage_aliases_available_2 is None:
            return None
        if storage_aliases_available_1 is None or storage_aliases_available_2 is None:
            return storage_aliases_available_1 or storage_aliases_available_2

        return {**storage_aliases_available_1, **storage_aliases_available_2}

    def add(self, slave_info: SlaveInfo2) -> "ClusterInfo":
        """Add Slave Info to Cluster Info."""
        return ClusterInfo(
            deployment_environment=self.deployment_environment,
            cluster_id=self.cluster_id,
            comment=self.comment,
            is_private=self.is_private,
            slave_count=self.slave_count + 1,
            gpu_model=list(
                {
                    *self.gpu_model,
                    *slave_info.gpu_model,
                }
            ),
            worker=self.worker.add(slave_info.worker),
            cpu_memory_mb=self.cpu_memory_mb.add(slave_info.cpu_memory_mb),
            gpu=self.gpu.add(slave_info.gpu),
            cpu=self.cpu.add(slave_info.cpu),
            have_access=self.have_access,
            allowed_projects=self.allowed_projects,
            active_jobs_counts=self.active_jobs_counts,
            storage_paths_available=ClusterInfo._merge_storage_paths(
                self.storage_paths_available, slave_info.storage_paths_available
            ),
            storage_aliases_available=ClusterInfo._merge_storage_aliases(
                self.storage_aliases_available, slave_info.storage_aliases_available
            ),
        )
