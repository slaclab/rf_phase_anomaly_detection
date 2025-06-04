import logging
from datetime import datetime
from multiprocessing import Queue
from multiprocessing import queues
from logging.handlers import QueueHandler

from typing import Optional


default_logging_kwargs = {
    'queue': None,
    'logger_name': None,
    'log_level': logging.DEBUG,
    'log_stdout': True
}


def run_logger_process(
        queue: Optional[Queue] = None,
        logger_name: Optional[str] = None,
        log_level: int = 10,
        log_stdout: bool = False
):
    # configure formatter
    t = datetime.strftime(datetime.now(), '%Y%m%d_%H%M%S')
    fmt_str = '%(asctime)s | %(name)-15s | %(levelname)-8s | %(message)s'
    dt_str = "%Y-%m-%dT%H:%M:%S"
    formatter = logging.Formatter(fmt=fmt_str, datefmt=dt_str)

    # create a logger
    name = logger_name if logger_name is not None else 'default'
    logger = logging.getLogger(name)
    # log all messages at this level or above
    logger.setLevel(log_level)
    if log_stdout:
        # configure a stream handler
        sch = logging.StreamHandler()
        sch.setFormatter(formatter)
        logger.addHandler(sch)

    if logger_name is not None:
        # configure a file handler
        cch = logging.FileHandler(f"{logger_name:s}_{t:s}.log")
        cch.setFormatter(formatter)
        logger.addHandler(cch)
    else:
        logger.addHandler(logging.NullHandler())

    # if isinstance(queue, queues.Queue):  # returns False
    if queue is not None:
        # run forever
        while True:
            # consume a log message, block until one arrives
            message = queue.get()
            # check for shutdown
            if message is None:
                break
            # log the message
            logger.handle(message)


def create_worker_logger(
        queue: Optional[Queue] = None,
        logger_name: Optional[str] = None,
        log_level: int = 0,
        log_stdout: bool = False  # for conformity with run_logger_process
) -> Optional[logging.Logger]:
    """same signature as run_logger_process"""
    name = logger_name if logger_name is not None else 'worker'
    # create a logger
    logger = logging.getLogger(name)

    # if isinstance(queue, queues.Queue):  # returns False
    if queue is not None:
        # add a handler that uses the shared queue
        logger.addHandler(QueueHandler(queue))
    # log all messages, debug and up
    logger.setLevel(log_level)
    # report initial message
    logger.info(f'Child process named "{name}" starting with log level {log_level}.')
    return logger
