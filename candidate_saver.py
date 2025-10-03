import os
import numpy as np

from mp_logging import create_worker_logger, default_logging_kwargs

from typing import Optional


class CandidateSaver:
    def __init__(
            self,
            directory: Optional[str] = None,
            logging_kwargs: Optional[dict] = default_logging_kwargs,
    ):
        """
        Saves anomalies in to json files in a directory.

        Parameters
        ----------
        directory
        logging_kwargs
        """
        self.directory = directory
        self.filename_prefix = "anomaly_candidate"
        self.filename_suffix = ".npz"

        self.logging_kwargs = logging_kwargs
        self.logging_kwargs["logger_name"] = "json_saver"
        self.logger = None

        self.file_counter = None

    def initialize(self):
        if self.directory is not None:
            self.logger = create_worker_logger(**self.logging_kwargs)
            self.logger.info("Initializing anomaly candidate saver")
            if os.path.exists(self.directory):
                self.logger.info(f"Saving anomalies to directory {self.directory:s}")
                self.file_counter = len(
                    [f for f in os.listdir(self.directory)
                     if f.startswith(self.filename_prefix) and f.endswith(self.filename_suffix)]
                )
            else:
                self.logger.info(f"Directory {self.directory:s} not found, creating it...")
                os.makedirs(self.directory, exist_ok=True)
                self.file_counter = 0

    def save_anomaly_candidate(self, anomaly: dict) -> None:
        if self.file_counter is None:
            self.initialize()
        if self.directory is not None:
            self.logger.debug(
                f"Saving anomaly candidate number {self.file_counter:d}"
            )
            fn = os.path.join(
                self.directory,
                f"{self.filename_prefix:s}_{self.file_counter:06d}{self.filename_suffix:s}"
            )
            np.savez(fn, **anomaly)
            self.file_counter += 1
