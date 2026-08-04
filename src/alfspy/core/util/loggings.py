import time
from logging import Logger, LogRecord
import logging.config
from typing import ContextManager, Final, Optional

DEFAULT_LOGGING_CONFIG: Final[dict] = {
    'version': 1,
    'disable_existing_loggers': True,
    'formatters': {
        'simple': {
            'format': '[%(asctime)s] %(message)s',
        }
    },
    'handlers': {
        'default': {
            'class': 'logging.StreamHandler',
            'level': 'DEBUG',
            'formatter': 'simple',
            'stream': 'ext://sys.stdout',
        }
    },
    'root': {
        'level': 'DEBUG',
        'handlers': ['default'],
    }
}

def apply_default_config() -> None:
    """
    Applies the default logging configuration.
    """
    logging.config.dictConfig(DEFAULT_LOGGING_CONFIG)

class LoggerStep(ContextManager):

    def __init__(self, logger: Logger, enter_msg: str, exit_msg: str = 'Done', visualize_step: bool = True) -> None:
        """
        A lightweight enterable logging object that logs an enter message and an exit message. The time passed between
        entering and exiting the context is appended to the exit message: ``'{exit_msg} [{time_in_ms} ms]'``.
        :param logger: The logger to be used for any logging.
        :param enter_msg: The enter message to be logged.
        :param exit_msg: The exit message to be logged.
        :param visualize_step: Whether to visualize the logging step (defaults to ``True``).
        """
        self.logger = logger
        self.enter_msg = enter_msg
        self.enter_time = None
        self.exit_msg = exit_msg
        self.exit_time = None
        if visualize_step:
            self.filter = logging.Filter()
            self.filter.filter = self._filter
            self.exit_msg = f'└ {exit_msg}'
        else:
            self.filter = None

    @staticmethod
    def _filter(record: LogRecord) -> bool:
        record.msg = f'│ {record.msg}'
        return True

    def __enter__(self) -> None:
        self.enter_time = time.time()
        self.logger.info(self.enter_msg)
        if self.filter is not None:
            self.logger.addFilter(self.filter)

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.filter is not None:
            self.logger.removeFilter(self.filter)
        self.exit_time = time.time()
        log_time = (self.exit_time - self.enter_time) * 1000
        self.logger.info(f'{self.exit_msg} [{log_time:.3f} ms]')


