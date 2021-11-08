import builtins
import json
from logging import getLogger
from logging.config import dictConfig
import os
import pkgutil
import sys
from typing import Optional

try:
    import colorlog

    _HAS_COLORLOG = True

except ImportError:

    _HAS_COLORLOG = False

logger = getLogger(__name__)


def _suppress_logging() -> None:
    """Suppresses logging from the current process."""
    getLogger().setLevel(60)


def _suppress_print() -> None:
    """Suppresses printing from the current process."""

    def ignore(*_objects, _sep=" ", _end="\n", _file=sys.stdout, _flush=False):
        pass

    builtins.print = ignore


def _load_config(colored: bool) -> dict:
    """Load static logging config file."""
    if colored and _HAS_COLORLOG:
        config_file = "logging_config/colored.json"
    else:
        config_file = "logging_config/normal.json"
    config_data = pkgutil.get_data(__name__, config_file)
    if config_data is not None:
        config = json.loads(config_data.decode("utf-8"))
    else:
        config = {}
    return config


def config_logging(enabled: bool = True, log_file: Optional[str] = None, colored: bool = True) -> None:
    """Config root logger.

    Args:
        enable: True if enable logging, False otherwise.
        log_file: Path to the log file.

    """

    config = _load_config(colored)

    if log_file:
        # user customized file location
        config["handlers"]["file_handler"]["filename"] = str(log_file)

    # config root logger
    dictConfig(config)

    # disable all logging and print for child process
    if enabled:
        logger.debug("Logging is enabled for process %d.", os.getpid())
    else:
        logger.debug("Logging is disable for process %d.", os.getpid())
        _suppress_logging()
        _suppress_print()
