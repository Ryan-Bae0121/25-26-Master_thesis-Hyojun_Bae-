"""GPULab config."""

from typing import Any, TypeVar

from pydantic import BaseModel
from pydantic_yaml import parse_yaml_file_as, parse_yaml_raw_as, to_yaml_file
from typing_extensions import deprecated


def discard_pem_certs(pem_content: str) -> str:
    """Discard PEM certificates from string."""
    res = ""
    include = True
    for line in pem_content.splitlines(keepends=True):
        if line.startswith("-----BEGIN CERTIFICATE-----"):
            include = False
        if include:
            res += line
        if line.startswith("-----END CERTIFICATE-----"):
            include = True
    return res


def discard_pem_privkeys(pem_content: str) -> str:
    """Discard PEM private keys from string."""
    res = ""
    include = True
    for line in pem_content.splitlines(keepends=True):
        if line.startswith(("-----BEGIN RSA PRIVATE KEY-----", "-----BEGIN PRIVATE KEY-----")):
            include = False
        if include:
            res += line
        if line.startswith(("-----END RSA PRIVATE KEY-----", "-----END PRIVATE KEY-----")):
            include = True
    return res


BCShape = TypeVar("BCShape", bound="BaseConfig")


class BaseConfig(BaseModel):
    """Base class for configs."""

    @classmethod
    def load_from_file(cls: type[BCShape], config_file: str) -> BCShape:
        """Load from YAML file."""
        return parse_yaml_file_as(cls, config_file)

    @classmethod
    def load_from_ymlstr(cls: type[BCShape], ymlstr: str) -> BCShape:
        """Load from YAML string."""
        return parse_yaml_raw_as(cls, ymlstr)

    def save_to_file(self, filename: str) -> None:
        """Save to YAML file."""
        to_yaml_file(filename, self)

    @deprecated("Use model_dump instead.")
    def save_to_dict(self) -> dict[str, Any]:
        """Save to dict."""
        return self.model_dump()

    @classmethod
    @deprecated("Use model_validate instead.")
    def load_from_dict(cls: type[BCShape], v: dict[str, Any]) -> BCShape:
        """Load from dict."""
        return cls.model_validate(v)
