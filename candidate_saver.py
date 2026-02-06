import os
import numpy as np
from datetime import datetime

from run_config import NANOSECS_IN_1_SEC
from mp_logging import create_worker_logger, default_logging_kwargs

from typing import Optional


class Saver:
    keep_prefix = "candidate"
    reject_prefix = "reject"
    filename_suffix = ".npz"
    logger_name = "numpy_saver"
    keep_verb = "positive"
    reject_verb = "negative"

    def __init__(
            self,
            directory: Optional[str] = None,
            logging_kwargs: Optional[dict] = default_logging_kwargs,
    ):
        """
        Saves candidates to numpy files in a directory.

        Parameters
        ----------
        directory : str, optional
            Where to save the files.  If None, it will save nothing.
        logging_kwargs : dict, optional
            See mp_logging documentation for the set of logging kwargs.
        """
        self.directory = directory

        self.logging_kwargs = logging_kwargs
        self.logging_kwargs["logger_name"] = self.logger_name
        self.logger = None

        self.file_counter = None

    def initialize(self):
        if self.directory is not None:
            self.logger = create_worker_logger(**self.logging_kwargs)
            self.logger.info("Initializing saver")
            if os.path.exists(self.directory):
                self.logger.info(f"Saving to directory {self.directory:s}")
                self.file_counter = len(
                    [f for f in os.listdir(self.directory)
                     if f.startswith(self.keep_prefix) and f.endswith(self.filename_suffix)]
                )
            else:
                self.logger.info(f"Directory {self.directory:s} not found, creating it...")
                os.makedirs(self.directory, exist_ok=True)
                self.file_counter = 0

    def save(self, anomaly: dict, reject: bool = False) -> None:
        if self.file_counter is None:
            self.initialize()
        if self.directory is not None:
            if not reject:  # not reject is an anomaly candidate to keep
                log_verb = self.keep_verb
                prefix = self.keep_prefix
            else:  # reject this anomaly candidate
                log_verb = self.reject_verb
                prefix = self.reject_prefix

            self.logger.debug(
                f"Saving {log_verb:s} number {self.file_counter:d}"
            )
            u = datetime.fromtimestamp(anomaly["candidate_timestamp"] / NANOSECS_IN_1_SEC)  # time in Pacific time
            date = str(u.date()).replace("-", "")
            time = str(u.time()).split('.')[0].replace(":", "")
            fn = os.path.join(
                self.directory,
                f"{prefix:s}_{date:s}_{time:s}_{self.file_counter:06d}{self.filename_suffix:s}"
            )
            np.savez_compressed(fn, **anomaly, allow_pickle=True)  # pickle is required to save objects
            self.file_counter += 1


class CandidateSaver(Saver):
    keep_prefix = "anomaly_candidate"
    reject_prefix = "rejected_candidate"
    logger_name = "candidate_saver"
    keep_verb = "candidate"
    reject_verb = "reject"


class AnomalySaver(Saver):
    keep_prefix = "confirmed_anomaly"
    reject_prefix = "rejected_anomaly"
    logger_name = "anomaly_saver"
    keep_verb = "anomaly"
    reject_verb = "reject"
