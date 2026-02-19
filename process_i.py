from multiprocessing import Manager

from process import CustomProcessObject
from mp_logging import default_logging_kwargs, create_worker_logger
from machine_interface.k2eg_instrument_portal import K2EGInstrumentPortal as InstrumentPortal
#from machine_interface.p4p_instrument_portal import P4PInstrumentPortal as InstrumentPortal

from typing import Optional, Any, TypedDict


class Message(TypedDict):
    method: str
    data: Any
    serialization: str


class ProcessI(CustomProcessObject):
    """
    Allows communication with the GUI through K2EG puts.
    The 'I' stands for instrumentation.
    """

    def __init__(
            self,
            queue: "Manager.Queue",
            logging_kwargs: Optional[dict] = default_logging_kwargs,
    ):
        self.queue = queue
        self.logging_kwargs = logging_kwargs
        self.logging_kwargs["logger_name"] = "Process_i"
        self.logger = None

        self.portal = None

    def __call__(self):
        if self.logger is None:
            self.logger = create_worker_logger(**self.logging_kwargs)
        if not isinstance(self.portal, InstrumentPortal):
            self.portal = InstrumentPortal(
                logging_kwargs=self.logging_kwargs.copy(),
            )

        # put data onto the queue at regular intervals
        with self.portal as portal:
            self.logger.info(f"InstrumentPortal.is_running: {portal.is_running}")
            while portal.is_running:
                # consume a log message, block until one arrives
                message = self.queue.get()
                # check for shutdown
                if message is None:
                    self.logger.info("Portal queue received signal to terminate")
                    break
                else:
                    check_and_pass_message(message, self.portal)
        self.logger.debug("Instrument portal has been closed")

        for handler in self.logger.handlers:
            handler.close()


def check_and_pass_message(
        message: dict[str, Any],
        portal: InstrumentPortal
):
    log_msg = ""

    if "method" not in message:
        log_msg += "Message missing 'method' key; "
    else:
        if not isinstance(message["method"], str):
            log_msg += "Message['method'] is not a string; "

    if "data" not in message:
        log_msg += "Message missing 'data' key; "

    # if "serialization" not in message:
    #     log_msg += "Message missing 'serialization' key; "
    # else:
    #     if not isinstance(message["method"], str):
    #         log_msg += "Message['serialization'] is not a string; "

    if log_msg == "":
        portal(**message)
    else:
        full_msg = "Put ignored due to improper message format: " + log_msg[:-2]
        portal.logger.warning(full_msg)