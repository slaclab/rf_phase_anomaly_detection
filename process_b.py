# standard library imports
import os
import time
from datetime import datetime
from multiprocessing import Manager
from queue import Empty
from typing import Optional

# 3rd party imports
import numpy as np

# local imports
from mp_logging import create_worker_logger, default_logging_kwargs
from process import CustomProcessObject
from buffer import Buffer, SAMPLES_PER_SECOND, BUFFER_DURATION_SEC, BUFFER_LENGTH
import k2eg_spoofer

# we care about windows where beam-checks fail only if longer than this length
TEMP_VIOLATION_LENGTH = SAMPLES_PER_SECOND * 90 # 90 seconds

class ProcessB(CustomProcessObject):
    """
    ProcessB consumes k2eg snapshots from queue_one and buffers 5 minutes of data per PV at 120hz.
    it performs beam checks and candidate window searcing, and forwards output to queue_two for ProcessC.
    """
    def __init__(self,
                queue_one: 'Manager.Queue',
                queue_two: 'Manager.Queue',
                pv_list: list[str],
                logging_kwargs: Optional[dict] = default_logging_kwargs
                ) -> None:
        self.queue_one = queue_one
        self.queue_two = queue_two

        self.pv_list = pv_list

        self.logging_kwargs = logging_kwargs
        self.logging_kwargs['logger_name'] = 'process_b'
        self.logger = None

        # holds up to 5 minutes of 120hz data (36000 points) per pv.
        self.buffer = Buffer(pv_list, BUFFER_LENGTH, logging_kwargs) # 3600 = 120hz * 60sec * 5mins

    def __call__(self) -> None:

        if self.logger is None:
            self.logger = create_worker_logger(**self.logging_kwargs)

        self.logger.debug(f"running process_b on {len(self.pv_list)} pvs")
        self.logger.debug("starting data processing loop...")

        while True:
            try:
                r = self.queue_one.get(timeout=0.05) # wait 50ms

                if r is None: # enqueuing a None should stop this process
                    break

                # need to be sure each iteration of this processing loop is <= 1 second
                # (new data comes each second from process_a, so data will pile-up if our processing takes over 1 second)
                start = time.perf_counter()

                # parse the k2eg snapshot and update buffer
                self.buffer.update(r)

                # process data in buffer and get stuff to pass to process_c and CoAD
                # valid_windows = self.buffer.get_valid_windows()
                # result = self.find_candidates()

                end = time.perf_counter()
                elapsed_ms = (end - start) * 1000
                self.logger.debug(f"process_b iteration took : {elapsed_ms:.2f} ms")
                if elapsed_ms > 1000: # have to be <= 1 sec
                    self.logger.warning(f"process_b iteration is slow!! : {elapsed_ms:.2f} ms")

                
                # placeholder: dummy data for ProcessC:
                fake_rf_input_tensor = np.random.rand(1, 1066).astype(np.float32)
                fake_bpm_input_tensor = np.random.rand(8, 1066).astype(np.float32)
                fake_pv_name = "fake_pv_name"

                # expand later with any more info we need to send to process_c
                fake_output_data = {
                    "timestamp": end,
                    "rf_input": fake_rf_input_tensor,
                    "bpm_input": fake_bpm_input_tensor,
                    "pv_name": fake_pv_name,
                }
                self.queue_two.put(fake_output_data)

            except Empty:
                continue

        self.logger.debug("shutting down process_b")

        for handler in self.logger.handlers:
            handler.close() 