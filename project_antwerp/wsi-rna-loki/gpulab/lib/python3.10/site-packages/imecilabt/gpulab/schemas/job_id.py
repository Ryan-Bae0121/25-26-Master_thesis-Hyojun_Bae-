"""Job ID validation functions."""

import re

from imecilabt_utils.validation_utils import is_valid_uuid

# Note: Job ID is a 36 char UUID (thus including the hyphens)
ILLEGAL_JOB_ID_CHARACTERS_REGEX = re.compile(r"[^0-9a-fA-F-]")

JOB_ID_LENGTH = 36


def is_partial_job_id(job_id: str) -> bool:
    """Check if string is at least a partial job id."""
    if re.search(ILLEGAL_JOB_ID_CHARACTERS_REGEX, job_id):
        return False
    return len(job_id) < JOB_ID_LENGTH


def is_full_job_id(job_id: str) -> bool:
    """Check if string is a full job id."""
    return not is_bad_job_id(job_id)


def is_bad_job_id(job_id: str) -> bool:
    """Check if job_id is certainly bad: too short, wrong chars, too long..."""
    return bool(
        len(job_id) != JOB_ID_LENGTH
        or re.search(ILLEGAL_JOB_ID_CHARACTERS_REGEX, job_id)
        or not is_valid_job_id(job_id)
    )


def is_valid_job_id(id: str) -> bool:
    """Check if job_id is a valid UUID."""
    return bool(is_valid_uuid(id))
