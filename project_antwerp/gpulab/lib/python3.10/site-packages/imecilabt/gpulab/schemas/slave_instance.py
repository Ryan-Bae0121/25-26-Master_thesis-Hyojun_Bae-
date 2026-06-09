"""Slave Instance."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import Field
from typing_extensions import deprecated

from imecilabt.gpulab.schemas.slave_info2 import SlaveInstanceBase

from .base import AwareDatetime, BaseSchema

#
# These enums and dataclasses are used by the supervisor.
#


class SlaveInstanceState(Enum):
    """State of a slave instance."""

    DEAD = 0  # no dir/files, or some critical files missing
    STOPPED = 1  # dir/files ok, but not running
    RUNNING = 2  # Currently running (and config does not have shutdown)
    STOPPING = 3  # Currently shutting down (running and config has shutdown)
    UPDATING = 4  # Updating config
    HANG = 5  # RUNNING, but watchdog alive not reported -> internal hang!
    MUST_DELETE = 6  # all files must be deleted (will result in DEAD, but separate explicit state for safety)


class ResourceState(Enum):
    """State of a resource (gpu, cpu)."""

    UNKNOWN = 0
    FREE_ALL = 1
    USED_OTHER = 2
    USED_DEAD = 3
    FREE_SELF = 4
    USED_SELF = 5
    CONFLICT = 6
    WAIT_AFTER_DEAD_CLAIM_CLEANUP = 10


class ResourceType(Enum):
    """Resource Type."""

    GPU = 0
    CPU = 1


class SlaveInstanceResource(BaseSchema):
    """Resource bound to a slave instance."""

    type: ResourceType
    id: int
    state: ResourceState


class SlaveInstance(SlaveInstanceBase):
    """Slave Instance."""

    name: str
    instance_id: str
    state: SlaveInstanceState

    aliases: list[str] = Field(default_factory=list)
    comment: str | None = None
    host: str | None = None

    pid: int | None = None
    cluster_id: int
    commit_id: str | None = None
    commit_date: AwareDatetime | None = None
    branch: str | None = None
    config: dict[str, Any] | None = None
    base_dir: str | None = None
    dir: str | None = None
    config_filename: str | None = None
    pid_file: str | None = None
    pid_create_time: AwareDatetime | None = None
    venv_name: str | None = None
    repo_dir: str | None = None
    instance_create_date: AwareDatetime | None = None

    updated: datetime | None = None
    resources: list[SlaveInstanceResource] = Field(default_factory=list)

    @deprecated("Use instance_id instead.")
    @property
    def hash_id(self) -> str:
        """Alias for backward compatibility."""
        return self.instance_id

    @deprecated("Use instance_id instead.")
    @property
    def hash_nick(self) -> str:
        """Alias for backward compatibility."""
        return self.instance_id[0:15]
