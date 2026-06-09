"""Proxy configuration."""

from imecilabt_utils.urn_util import URN, always_optional_urn

from imecilabt.gpulab.util.gpulab_config import BaseConfig


class ProxyConfig(BaseConfig):
    """Config of proxy username mapping.

    Example:
    -------
    proxy_username_prefix:
        wall2.ilabt.iminds.be: ''  # wall2.ilabt.iminds.be usernames are NOT prefixed. Example: 'wvdemeer'->'wvdemeer'
        ilabt.imec.be: 'fff'       # ilabt.imec.be usernames are prefixed with 'fff', example: 'wvdemeer'->'fffwvdemeer'

    """

    proxy_username_prefix: dict[str, str]

    def find_proxy_username(self, user_urn: str | URN) -> str | None:
        """Determine proxy username for user URN."""
        u = always_optional_urn(user_urn)
        if u and u.name and u.authority and u.authority in self.proxy_username_prefix:
            return f"{self.proxy_username_prefix[u.authority]}{u.name}"
        return None
