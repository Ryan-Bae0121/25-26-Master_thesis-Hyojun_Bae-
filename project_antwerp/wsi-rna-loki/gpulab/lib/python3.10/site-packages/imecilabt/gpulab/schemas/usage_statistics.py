"""Job Usage Statistics.

2 separate usage statistics:
  "cpu" -> CpuUsageStatistics -> not just CPU, but everything but GPU -> reported by docker etc
  "gpu"  -> GPUUsageStatistics -> GPU -> reported by nv libraries
Combined in GPULabUsageStatistics
"""

import datetime
from typing import Any

from pydantic import ConfigDict, Field, computed_field, model_validator
from pydantic.alias_generators import to_camel
from typing_extensions import deprecated

from imecilabt.gpulab.util.enum import CaseInsensitiveEnum

from .base import AwareDatetime, BaseSchema


def to_milli_watt_friendly_camel(snake: str) -> str:
    """Camel case with exception for mW(h)-related fields."""
    if snake.endswith("_mWh_used"):
        return to_camel(snake[: -len("_mWh_used")]) + "mWhUsed"
    if snake.endswith("_mW_used"):
        return to_camel(snake[: -len("_mW_used")]) + "mWUsed"
    return to_camel(snake)


model_config_with_milli_watt_friendly_camel = ConfigDict(
    alias_generator=to_milli_watt_friendly_camel,
    # Because we have @computed fields, we need to ignore them when validating.
    extra="ignore",
    frozen=True,
    populate_by_name=True,
    from_attributes=True,
    use_enum_values=True,
)


class ContainerUsageStatistics(BaseSchema):
    """Container Usage Statistics."""

    first_time: datetime.datetime
    last_time: datetime.datetime
    agg_period_ns: int
    cpu_count: int
    cpu_usage: float  # in nr CPU's, so between 0 and cpu_count
    cpu_usage_total_ns: int
    cpu_usage_kernelmode_ns: int
    cpu_usage_usermode_ns: int
    max_pid_count: int
    mem_limit_byte: int
    mem_usage_byte: int
    mem_max_usage_byte: int
    network_rx_byte: int
    network_tx_byte: int
    # power measurements
    #    (None if not (yet) available/implemented)
    #    (and for backward compatibility default is 0.0)
    #
    # Power used by the host machine itself (not CPU or GPU) during period between first_time and last_time.
    #   Proportional to number of cpu's of the total used.
    #   (Might be proportional to something else in future versions, but idea is the same.)
    host_total_mWh_used: int | None = None  # total Wh (Watt hour)
    host_average_mW_used: int | None = None  # average W (Watt)
    #
    # Power used by all CPU's of this job, during this job.
    cpu_total_mWh_used: int | None = None  # total Wh (Watt hour)
    cpu_average_mW_used: int | None = None  # average W (Watt)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_mWh_used(self) -> int | None:
        """Power used by all non-GPU resources of this job during this job, in mWh.

        (= host_* and cpu_* power stats added)
        """
        if self.host_total_mWh_used or self.cpu_total_mWh_used:
            return (self.host_total_mWh_used or 0) + (self.cpu_total_mWh_used or 0)
        return None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def average_mW_used(self) -> float | None:
        """Average power used by all non-GPU resources of this job during this job, in mW."""
        if self.host_average_mW_used or self.cpu_average_mW_used:
            return (self.host_average_mW_used or 0.0) + (self.cpu_average_mW_used or 0.0)
        return None

    def is_invalid(self) -> bool:
        """Check if statistics are invalid."""
        return (
            self.agg_period_ns <= 0
            or self.cpu_count <= 0
            or self.first_time <= datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)
        )

    def to_timeseriesdb_entry(self) -> dict[str, Any]:
        """Transform to entry for timeseries database."""
        # self.total_mWh_used and self.average_mmW_used are not included in timeseries DB!
        return {
            "sample_period_ns": self.agg_period_ns,
            "pid_count": self.max_pid_count,
            "cpu_usage_total_ns": self.cpu_usage_total_ns,
            "cpu_usage_kernelmode_ns": self.cpu_usage_kernelmode_ns,
            "cpu_usage_usermode_ns": self.cpu_usage_usermode_ns,
            "mem_usage_byte": self.mem_usage_byte,
            # 'mem_max_usage_byte': self.mem_max_usage_byte,
            "mem_limit_byte": self.mem_limit_byte,
            "cpu_usage": self.cpu_usage,
            "cpu_usage_percent_all": ((self.cpu_usage * 100.0) / self.cpu_count) if self.cpu_count > 0 else -1,  # float
            "cpu_count": self.cpu_count,
            "network_rx_byte": self.network_rx_byte,
            "network_tx_byte": self.network_tx_byte,
            "host_total_mWh_used": self.host_total_mWh_used,
            "host_average_mW_used": self.host_average_mW_used,
            "cpu_total_mWh_used": self.cpu_total_mWh_used,
            "cpu_average_mW_used": self.cpu_average_mW_used,
        }

    @classmethod
    def invalid(cls) -> "ContainerUsageStatistics":
        """Return invalid ContainerUsageStatistics instance."""
        return ContainerUsageStatistics(
            first_time=datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc),
            last_time=datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc),
            agg_period_ns=-1,
            cpu_count=-1,
            cpu_usage=-1.0,
            cpu_usage_total_ns=-1,
            cpu_usage_kernelmode_ns=-1,
            cpu_usage_usermode_ns=-1,
            max_pid_count=-1,
            mem_limit_byte=-1,
            mem_usage_byte=-1,
            mem_max_usage_byte=-1,
            network_rx_byte=-1,
            network_tx_byte=-1,
            host_total_mWh_used=None,
            host_average_mW_used=None,
            cpu_total_mWh_used=None,
            cpu_average_mW_used=None,
        )

    model_config = model_config_with_milli_watt_friendly_camel


class GpuUsageStatistics(BaseSchema):
    """GPU Usage Statistics."""

    gpu_count: int
    average_utilization: float  # in number of GPU's, so between 0 and gpu_count
    average_mem_utilization: float  # in number of GPU's, so between 0 and gpu_count
    # power measurements
    #    (None if not (yet) available/implemented)
    #    (and for backward compatibility default is 0.0)
    #
    # Power used by all GPU's during the relevant period.
    total_mWh_used: int | None = None  # total Wh (Watt hour)
    average_mW_used: int | None = None  # average W (Watt)

    @classmethod
    def empty(cls) -> "GpuUsageStatistics":
        """GPU Usage Statistics with zero values."""
        return GpuUsageStatistics(
            gpu_count=0,
            average_utilization=0.0,
            average_mem_utilization=0.0,
            total_mWh_used=0,
            average_mW_used=0,
        )

    model_config = model_config_with_milli_watt_friendly_camel


@deprecated("Deprecated class.")
class WasteStat(BaseSchema):
    """Waste stat. Deprecated."""

    # These stats are all averaged over time
    cpu_usage_perc: float  # percent of all CPUs used (average)
    cpu_memory_perc: float  # percent of CPU mem used (average)
    wasted: bool  # is this stat considered "waste of resources"?
    gpu_usage_perc: float | None = None  # percent of all GPU used (average)
    gpu_active_processes_per_gpu: float | None = None  # number of active processes per GPU (average)
    wasted_reason: list[str] = Field(default_factory=list)

    @classmethod
    def empty(cls) -> "WasteStat":
        """Empty WasteStat."""
        return cls(
            cpu_usage_perc=0.0,
            cpu_memory_perc=0.0,
            wasted=False,
            gpu_usage_perc=None,
            gpu_active_processes_per_gpu=None,
        )


class WasteVerdict(CaseInsensitiveEnum):
    """Waste Verdict."""

    GOOD = "Good"
    """Job is not wasting resources (yet, for running jobs)"""

    UNCERTAIN = "Uncertain"
    """Job might be wasting resources, or might not"""

    WASTE = "Waste"
    """Job is wasting resources"""
    SHORT = "Short"
    """Job was too short to verdict. It's OK if short test/failed jobs waste resources."""
    UNDECIDED = "Undecided"
    """Too soon to tell if job wastes resources"""


class WasteFlag(CaseInsensitiveEnum):
    """Waste Flag."""

    RED = "Red Flag"
    """Certainly unacceptable resource waste"""
    ORANGE = "Orange Flag"
    """Quite likely unacceptable resource waste"""
    YELLOW = "Yellow Flag"
    """Resource waste, but probably not unacceptable"""


class WasteFlagInfo(BaseSchema):
    """Waste Flag Info."""

    flag: WasteFlag
    explanation: str


class WasteReview(BaseSchema):  # AKA WasteReport2, which replaces the legacy WasteReport
    """Waste Report."""

    updated: AwareDatetime  # date in RFC3339 format
    final: bool
    """This is the final report, as the job has finished. It can include details intermediate reports don't."""
    verdict: WasteVerdict
    """Is this job wasting resources, or not?"""
    flags: list[WasteFlagInfo] = Field(default_factory=list)
    """Why is this job wasting resources? (empty if not wasting)"""
    debug: dict[str, bool | int | float | str] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def ignore_deprecated_verdict_explanation(cls, data: Any) -> Any:
        """We ignore and remove the deprecated verdictExplanation."""
        if isinstance(data, dict):
            if "verdictExplanation" in data:
                del data["verdictExplanation"]
            if "verdict_explanation" in data:
                del data["verdict_explanation"]
        return data

    # This method has drawbacks. We really just want to allow it in incoming data, but completely remove it
    # deprecated__verdict_explanation: Any = Field(
    #     None, alias="verdictExplanation", exclude=True, deprecated=True
    # )


class GPULabUsageStatistics(BaseSchema):
    """GPULab Usage Statistics."""

    container_statistics: ContainerUsageStatistics
    gpu_statistics: GpuUsageStatistics

    # Power measurements
    #    (None if not (yet) available/implemented)
    #    (and for backward compatibility default is 0.0)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_mWh_used(self) -> int | None:
        """Total mWh used.

        Power used by all resource of this job (GPU + CPU + host) during period between
        container_statistics.first_time and container_statistics.last_time.
        """
        if self.container_statistics.total_mWh_used or self.gpu_statistics.total_mWh_used:
            return (self.container_statistics.total_mWh_used or 0) + (self.gpu_statistics.total_mWh_used or 0)
        return None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def average_mW_used(self) -> float | None:
        """Average mW used."""
        if self.container_statistics.average_mW_used or self.gpu_statistics.average_mW_used:
            return (self.container_statistics.average_mW_used or 0.0) + (self.gpu_statistics.average_mW_used or 0.0)
        return None

    model_config = model_config_with_milli_watt_friendly_camel


class GpuInfo(BaseSchema):
    """GPU Information."""

    index: int = Field(..., examples=[0])
    uuid: str = Field(..., examples=["GPU-8a56a4bc-e184-a047-2620-be19fdf913d5"])
    serial: str = Field(..., examples=["N/A"])
    name: str = Field(..., examples=["GeForce GTX 980"])
    brand: str = Field(..., examples=["GeForce"])
    minor_number: int = Field(..., examples=[0])
    board_id: int = Field(..., examples=[768])
    bridge_chip_info: str = Field(..., examples=["N/A"])
    is_multi_gpu_board: bool = Field(..., examples=[False])
    max_pcie_link_generation: int = Field(..., examples=[3])
    max_pcie_link_width: int = Field(..., examples=[16])
    vbios_version: str = Field(..., examples=["84.04.31.00.29"])


class GpuOverview(BaseSchema):
    """GPU Information Overview."""

    cuda_version_full: str = Field(..., examples=["10.2.0"], description="Retrieved from resource manager")
    cuda_version_int: int = Field(..., examples=[10020], description="Retrieved from nv utils")
    cuda_version_major: int = Field(..., examples=[10])
    cuda_version_minor: int | float = Field(..., examples=[2, 2.0])
    driver_version: str = Field(..., examples=["440.33.01"])
    nvml_version: str = Field(..., examples=["10.440.33.01"])
    gpus: list[GpuInfo]
