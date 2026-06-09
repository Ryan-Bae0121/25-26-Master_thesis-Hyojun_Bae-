"""GPULab logging handler."""

import logging
from logging import Handler

from imecilabt.gpulab.schemas.master import Master


class MasterJobEventLogHandler(Handler):
    """A handler class which writes logging records.

    It appropriately formats them and sends them to the common JobEvent handler. (which will write them to the DB).
    """

    terminator = "\n"

    def __init__(self, master: Master, job_id: str) -> None:
        """Initialize the handler.

        If stream is not specified, sys.stderr is used.
        """
        Handler.__init__(self)
        if not job_id:
            msg = "job_id is mandatory"
            raise RuntimeError(msg)
        self.job_id = job_id
        self.master = master

    def flush(self) -> None:
        """Flushes the stream."""
        # self.acquire()
        # try:
        #     if self.stream and hasattr(self.stream, "flush"):
        #         self.stream.flush()
        # finally:
        #     self.release()

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a record.

        If a formatter is specified, it is used to format the record.
        The record is then written to the stream with a trailing newline.  If
        exception information is present, it is formatted using
        traceback.print_exception and appended to the stream.  If the stream
        has an 'encoding' attribute, it is used to determine how to do the
        output to the stream.
        """
        try:
            msg = self.format(record)
            # job_event_type = logginglevel_to_jobeventype(record.levelno)
            self.master.register_logging_event(self.job_id, record.levelno, msg)
        except Exception:
            self.handleError(record)


def _job_log_handler(master: Master, job_id: str) -> Handler:
    return MasterJobEventLogHandler(master, job_id)


_default_handler = None


def setDefaultLogHandler(handler: Handler) -> None:
    """Set default logging handler."""
    global _default_handler  # noqa: PLW0603
    _default_handler = handler


def getLogger(name: str) -> logging.Logger:
    """Get logger."""
    logger = logging.getLogger(name)
    global _default_handler  # noqa: PLW0602
    if _default_handler:
        logger.addHandler(_default_handler)
    return logger


_logger_cache: dict[str, logging.Logger] = {}


def getJobLogger(job_id: str, master: Master) -> logging.Logger:
    """Get job logger."""
    if job_id in _logger_cache:
        logger = _logger_cache[job_id]
    else:
        _logger_cache[job_id] = logging.getLogger("job" + job_id)
        logger = _logger_cache[job_id]
        logger.addHandler(_job_log_handler(master, job_id))
    return logger
