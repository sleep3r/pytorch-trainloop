import logging
from contextlib import nullcontext
from typing import ContextManager

from rich.console import Console
from rich.traceback import install
from rich.logging import RichHandler

from config import DLConfig
from utils.distributed import get_dist_info

logger_initialized = {}
install(show_locals=False)


def get_logger(name, cfg: DLConfig = None, meta: dict = None, log_level=logging.INFO):
    """
    Initialize and get a logger by name.
    If the logger has not been initialized, this method will initialize the
    logger by adding one or two handlers, otherwise the initialized logger will
    be directly returned. During initialization, a StreamHandler will always be
    added. If `log_file` is specified and the process rank is 0, a FileHandler
    will also be added.

    Args:
        name (str): Logger name.
        cfg (DLConfig): config;
        meta (dict): meta info dict;
        log_level (int): The logger level. Note that only the process of
            rank 0 is affected, and other processes will set the level to
            "Error" thus be silent most of the time.
    Returns:
        logging.Logger: The expected logger.
    """
    if cfg is not None:
        log_level = cfg.training.log_level

    logger = logging.getLogger(name)

    if name in logger_initialized:
        return logger
    # handle hierarchical names
    # e.g., logger "a" is initialized, then logger "a.b" will skip the
    # initialization since it is a child of "a".
    for logger_name in logger_initialized:
        if name.startswith(logger_name):
            return logger

    logger.propagate = False

    handlers = [RichHandler()]

    rank, world_size = get_dist_info()

    # only rank 0 will add a FileHandler
    if rank == 0 and meta is not None:
        log_file = meta["exp_dir"] / 'run.log'
        # Here, the default behaviour of the official logger is 'a'. Thus, we
        # provide an interface to change the file mode to the default
        # behaviour.
        file_handler = logging.FileHandler(log_file, "w")
        handlers.append(file_handler)

    FORMAT = "%(message)s"
    formatter = logging.Formatter(FORMAT)
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)

    if rank == 0:
        logger.setLevel(log_level)
    else:
        logger.setLevel(logging.ERROR)

    logger_initialized[name] = True
    return logger


def print_log(msg, logger=None, level=logging.INFO):
    """
    Print a log message.

    Args:
        msg (str): The message to be logged.
        logger (logging.Logger | str | None): The logger to be used.
            Some special loggers are:
            - "silent": no message will be printed.
            - other str: the logger obtained with `get_root_logger(logger)`.
            - None: The `print()` method will be used to print log messages.
        level (int): Logging level. Only available when `logger` is a Logger object or "root".
    """
    if logger is None:
        print(msg)
    elif isinstance(logger, logging.Logger):
        logger.log(level, msg)
    elif logger == 'silent':
        pass
    elif isinstance(logger, str):
        _logger = get_logger(logger)
        _logger.log(level, msg)
    else:
        raise TypeError(f'logger should be either a logging.Logger object, str, silent or None, but got {type(logger)}')


def status(text: str) -> ContextManager:
    console = Console()
    rank, world_size = get_dist_info()

    if rank == 0:
        return console.status(text)
    else:
        return nullcontext()
