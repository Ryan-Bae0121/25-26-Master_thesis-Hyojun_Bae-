"""GPULab Extended Docker Image name."""

import re

from pydantic import model_validator
from pydantic.dataclasses import dataclass

# from https://docs.docker.com/engine/reference/commandline/tag/:
# Name components may contain lowercase letters, digits and separators.
# A separator is defined as a period, one or two underscores, or one or more dashes.
# A name component may not start or end with a separator.
# A tag name must be valid ASCII and may contain lowercase and uppercase letters,
# digits, underscores, periods and dashes. A tag name may not start with a period
# or a dash and may contain a maximum of 128 characters.

DOCKER_IMAGE_PATTERN = (
    r"^((?P<username>\w+):(?P<password>\w+)@)?"
    r"(?P<fullname>(?P<repository>[\w.\-_]+((?::\d+|)(?=\/[a-z0-9._-]+\/[a-z0-9._-]+))|)(?:\/|)"
    r"(?P<image>[a-z0-9.\-_]+(?:\/[a-z0-9.\-_]+|)+))(:(?P<tag>[a-zA-Z0-9_][a-zA-Z0-9_.-]{0,127}))?$"
)
DOCKER_IMAGE_REGEX = re.compile(DOCKER_IMAGE_PATTERN)


@dataclass
class DockerImageName:
    """GPULab uses an extended format for docker images, which includes user/password authentication details.

    This class provides easy access to all relevant parts.

    Official docker base image/tag reference, see https://docs.docker.com/engine/reference/commandline/tag/
    """

    image: str  # "image" in docker terminology: includes private repo DNS (but not tag)
    tag: str = "latest"
    user: str | None = None
    password: str | None = None

    @model_validator(mode="after")
    def _username_and_password(self) -> "DockerImageName":
        if (self.user is None) != (self.password is None):
            msg = "Both username and password are required, or neither can be present."
            raise ValueError(msg)
        return self

    # noinspection PyMethodFirstArgAssignment
    @classmethod
    def from_str(cls, image: str) -> "DockerImageName":
        """Parse string."""
        assert image is not None, "image is None"
        assert isinstance(image, str), f"image is not a str but a {type(image)}"
        assert image.strip(), f"image is empty: {image!r}"
        match = DOCKER_IMAGE_REGEX.match(image)

        if not match:
            msg = "Not a valid docker image name."
            raise ValueError(msg)

        return DockerImageName(
            image=match.group("fullname"),
            tag=match.group("tag") or "latest",
            user=match.group("username"),
            password=match.group("password"),
        )

    @property
    def has_auth(self) -> bool:
        """Does the image have authentication details."""
        return self.user is not None

    def __str__(self) -> str:
        """Return string representation."""
        if self.has_auth:
            return f"{self.user}:{self.password}@{self.image}:{self.tag}"

        return f"{self.image}:{self.tag}"

    @property
    def includes_registry(self) -> bool:
        """Does the image live in a specific registry."""
        # from https://docs.docker.com/engine/reference/commandline/tag/:
        # An image name is made up of slash-separated name components, optionally prefixed by a registry hostname.
        # The hostname must comply with standard DNS rules, but may not contain underscores.
        # If a hostname is present, it may optionally be followed by a port number in the format :8080.
        # If not present, the command uses Docker’s public registry located at registry-1.docker.io by default.

        # So we assume any dot is part of the DNS hostname of the registry
        return "." in self.image

    @property
    def registry(self) -> str:
        """The registry in which the image lives."""
        if self.includes_registry:
            first_slash = self.image.index("/")
            return self.image[:first_slash]
        return "registry-1.docker.io"  # public dockerhub registry

    @property
    def image_without_registry(self) -> str:
        """The image name without the registry."""
        if self.includes_registry:
            first_slash = self.image.index("/")
            return self.image[first_slash + 1 :]
        return self.image

    @property
    def image_with_tag(self) -> str:
        """Image with tag."""
        return f"{self.image}:{self.tag}"
