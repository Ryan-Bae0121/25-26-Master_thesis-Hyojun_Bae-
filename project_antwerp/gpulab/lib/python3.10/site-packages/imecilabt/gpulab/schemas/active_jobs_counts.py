"""Active Job Counts."""

from .base import BaseSchema


class ActiveJobsCounts(BaseSchema):
    """Return the number of jobs in any of the 'active' job states.

    Can be used for both clusters and slaves.
    """

    onhold: int
    queued: int
    assigned: int
    starting: int
    running: int
    musthalt: int
    halting: int

    @classmethod
    def zero(cls) -> "ActiveJobsCounts":
        """Return an ActiveJobsCounts with all counters on zero."""
        return cls(
            onhold=0,
            queued=0,
            assigned=0,
            starting=0,
            running=0,
            musthalt=0,
            halting=0,
        )
