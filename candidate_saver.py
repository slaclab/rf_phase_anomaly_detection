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
        self.root_directory = directory
        self.directory = directory

        self.logging_kwargs = logging_kwargs
        self.logging_kwargs["logger_name"] = self.logger_name
        self.logger = create_worker_logger(**self.logging_kwargs)
        self.logger.info(f"Initializing {str(self)}")

        self.file_counter = None

    def __str__(self):
        return 'Saver'

    def initialize(self, directory: str) -> None:
        """
        This method is meant to be called by self.save to keep self.directory
        pointing to a daily directory.
        """
        if directory == self.directory:
            # on first run, self.directory = self.root_directory, but directory
            # will be self.directory/date, so this will pass
            return

        if os.path.exists(directory):
            self.logger.info(f"Saving to directory {directory:s}")
            self.file_counter = len(
                [f for f in os.listdir(directory)
                 if f.startswith(self.keep_prefix) and f.endswith(self.filename_suffix)]
            )
        else:
            self.logger.info(f"Directory {directory:s} not found, creating it...")
            os.makedirs(directory, exist_ok=True)
            self.file_counter = 0

        self.directory = directory

    def save(self, anomaly: dict, reject: bool = False) -> None:
        if not reject:  # not reject is an anomaly candidate to keep
            log_verb = self.keep_verb
            prefix = self.keep_prefix
        else:  # reject this anomaly candidate
            log_verb = self.reject_verb
            prefix = self.reject_prefix
        if self.root_directory is not None:
            u = datetime.fromtimestamp(anomaly["candidate_timestamp"] / NANOSECS_IN_1_SEC)  # time in Pacific time
            date = str(u.date()).replace("-", "")
            time = str(u.time()).split('.')[0].replace(":", "")
            self.initialize(directory=os.path.join(self.root_directory, date))

            self.logger.debug(f"Saving {log_verb:s} number {self.file_counter:d}")
            fn = f"{prefix:s}_{date:s}_{time:s}_{self.file_counter:06d}{self.filename_suffix:s}"
            fp = os.path.join(self.directory, fn)
            np.savez_compressed(fp, **anomaly, allow_pickle=True)  # pickle is required to save objects
            self.file_counter += 1
        else:
            self.logger.debug(f"No directory specified, not saving {log_verb:s}")


class CandidateSaver(Saver):
    keep_prefix = "anomaly_candidate"
    reject_prefix = "rejected_candidate"
    logger_name = "candidate_saver"
    keep_verb = "candidate"
    reject_verb = "reject"

    def __str__(self):
        return 'CandidateSaver'


class AnomalySaver(Saver):
    keep_prefix = "confirmed_anomaly"
    reject_prefix = "rejected_anomaly"
    logger_name = "anomaly_saver"
    keep_verb = "anomaly"
    reject_verb = "reject"

    def __str__(self):
        return 'AnomalySaver'
