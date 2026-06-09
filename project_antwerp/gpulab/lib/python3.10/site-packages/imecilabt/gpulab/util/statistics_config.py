"""Statistics configuration."""

from pydantic import BaseModel, Field

from imecilabt.gpulab.util.gpulab_config import BaseConfig


class StatisticsConfig(BaseConfig):
    """Statistics configuration."""

    class ClickhouseStats(BaseModel):
        """Clickhouse configuration."""

        active: bool = Field(True, deprecated=True)
        host: str
        username: str
        password: str
        database: str
        native_port: int = 9440  # 9000 if insecure
        http_port: int = 8443  # 8123 if insecure
        secure: bool = True

    clickhouse_stats: ClickhouseStats | None
