"""GPULab Master.

Exposes all functionality of JobController, that is needed by the slave.
"""

from abc import ABC, abstractmethod

from imecilabt.gpulab.schemas.job2 import (
    Job as Job2,
)
from imecilabt.gpulab.schemas.job2 import (
    JobPortMapping as Job2PortMapping,
)
from imecilabt.gpulab.schemas.job2 import (
    JobStateResources as Job2StateResources,
)
from imecilabt.gpulab.schemas.job2 import (
    JobStatus as Job2Status,
)
from imecilabt.gpulab.schemas.job_filter3 import JobFilter3
from imecilabt.gpulab.schemas.slave_info2 import SlaveInfo2
from imecilabt.gpulab.schemas.usage_statistics import (
    GPULabUsageStatistics,
    GpuOverview,
)


class Master(ABC):
    """GPULab Master."""

    @abstractmethod
    def find_jobs3(self, job_filter: JobFilter3, page: int = 1, page_size: int = 10) -> list[Job2]:
        """Find jobs matching the job filter."""

    @abstractmethod
    def update_job_status(
        self, job_id: str, target_job_state: Job2Status, *, onlyif_current_state: Job2Status | None = None
    ) -> None:
        """Update Job status of a job."""

    @abstractmethod
    def init_job_state_resources(self, job_id: str, resources: Job2StateResources) -> None:
        """Set Job state resources."""

    @abstractmethod
    def init_job_state_resources_port_mapping(self, job_id: str, port_mappings: list[Job2PortMapping]) -> None:
        """Set Job port mapping."""

    @abstractmethod
    def init_job_state_resources_gpu_details(self, job_id: str, gpu_details: GpuOverview) -> None:
        """Set Job GPU details."""

    @abstractmethod
    def init_job_state_final_usage_statistics(self, job_id: str, final_usage_statistics: GPULabUsageStatistics) -> None:
        """Set Job final usage statistics."""

    @abstractmethod
    def get_job(self, job_id: str) -> Job2:
        """Get Job by id."""

    @abstractmethod
    def append_to_log(self, job_id: str, extra_content: bytes | str) -> None:
        """Append to job log."""

    # Predefined logging levels are ints mapping to: CRITICAL, ERROR, WARNING, INFO, DEBUG
    @abstractmethod
    def register_logging_event(self, job_id: str, level: int, msg: str, *, only_if_not_exists: bool = False) -> None:
        """Log job event."""

    @abstractmethod
    def report_slave_info(self, slave_info: SlaveInfo2) -> None:
        """Report current slave info."""
