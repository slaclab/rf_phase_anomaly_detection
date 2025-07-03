# standard library imports
import time
from multiprocessing import Manager
from queue import Empty
from typing import Optional, List

# 3rd party imports
import numpy as np

# local imports
from mp_logging import create_worker_logger, default_logging_kwargs
from process import CustomProcessObject
from buffer import Buffer
from anomaly_candidate import AnomalyCandidate, CandidateBucket, find_fast_index, find_most_anomalous_rf_station
from beam_check_config import SAMPLES_PER_SECOND, BUFFER_LENGTH, BPM_NAMES, RF_PV_NAMES

# we care about windows where beam-checks fail only if longer than this length
TEMP_VIOLATION_LENGTH = SAMPLES_PER_SECOND * 90  # 90 seconds


class ProcessB(CustomProcessObject):
    """
    ProcessB consumes k2eg snapshots from queue_one and buffers 5 minutes of data per PV at 120hz.
    it performs beam checks and candidate window searcing, and forwards output to queue_two for ProcessC.
    """

    def __init__(
        self,
        queue_one: "Manager.Queue",
        queue_two: "Manager.Queue",
        pv_list: list[str],
        logging_kwargs: Optional[dict] = default_logging_kwargs,
    ) -> None:
        self.queue_one = queue_one
        self.queue_two = queue_two

        self.pv_list = pv_list

        self.logging_kwargs = logging_kwargs
        self.logging_kwargs["logger_name"] = "process_b"
        self.logger = None

        # holds up to 5 minutes of 120hz data (36000 points) per pv.
        self.buffer = Buffer(pv_list, BUFFER_LENGTH, logging_kwargs)  # 3600 = 120hz * 60sec * 5mins
        # holds anomaly candidates
        self.candidate_bucket = CandidateBucket()

    def __call__(self) -> None:
        if self.logger is None:
            self.logger = create_worker_logger(**self.logging_kwargs)

        self.logger.debug(f"running process_b on {len(self.pv_list)} pvs")
        self.logger.debug("starting data processing loop...")

        while True:
            try:
                r = self.queue_one.get(timeout=0.05)  # wait 50ms
            except Empty:
                continue

            if r is None:  # enqueuing a None should stop this process
                break

            # need to be sure each iteration of this processing loop is <= 1 second
            # (new data comes each second from process_a, so data will pile-up if our processing takes over 1 second)
            start = time.perf_counter()

            # parse the k2eg snapshot and update buffer
            index_change, length_of_update = self.buffer.update(r)
            # move the indexes of the previously found candidates
            self.candidate_bucket.update_slow_indexes(index_change)
            # add new candidates to the bucket
            new_candidates: List[AnomalyCandidate] = self.buffer.find_candidates(look_back_this_far=length_of_update)
            for candidate in new_candidates:
                self.candidate_bucket.put(candidate)

            # check for candidates ready for process C
            while self.candidate_bucket.oldest_candidate_slow_index <= self.buffer.index - 5 * SAMPLES_PER_SECOND:
                candidate = self.candidate_bucket.get()  # get the oldest candidate

                # Calculate number of samples to look back from the slow trigger index
                lookback_seconds = 5
                lookback = lookback_seconds * SAMPLES_PER_SECOND

                buffer_index = self.buffer.index
                start = max(0, candidate.slow_index - lookback)
                end = min(buffer_index, candidate.slow_index)

                # Get the bpm_score_1 values for the lookback window
                bpm_score_1 = self.buffer.get("bpm_score_1", start, end)

                # find the fast trigger
                fast_index = find_fast_index(
                    bpm_score_1,
                    slow_index=candidate.slow_index,
                    window_size=20,  # should we put this in config?
                    start=start,
                    lookback_seconds=5,
                )
                candidate.fast_index = fast_index
                fast_time = self.buffer.get("pv_timestamp_ns", fast_index, fast_index + 1)[0]

                # find the most anomalous rf station
                window_size = 20
                min_index = max(0, candidate.slow_index - window_size)
                window = np.stack(
                    [self.buffer.get(pv, min_index, candidate.slow_index) for pv in RF_PV_NAMES], axis=1
                )  # (window_size, num_rf_pvs)
                most_anomalous_rf_pv_name, deviation_score, system_level_flag = find_most_anomalous_rf_station(
                    window,
                    rf_pv_names=RF_PV_NAMES,
                    phas_thresh=2.5,
                )  # What can we do with deviation score and flag?

                data_window = candidate.window_slice
                rf_input: np.array = self.buffer.get(most_anomalous_rf_pv_name, data_window[0], data_window[1]).copy()
                bpm_input = []
                for pv_name in BPM_NAMES:
                    bpm_input.append(self.buffer.get(pv_name, data_window[0], data_window[1]).copy())
                # make a candidate to send to process C
                cand = {
                    "timestamp": fast_time,
                    "rf_input": rf_input,
                    "bpm_input": np.vstack(bpm_input),
                    "rf_pv_name": most_anomalous_rf_pv_name,
                }
                self.queue_two.put(cand)

            # process data in buffer and get stuff to pass to process_c and CoAD,
            # do here...

            end = time.perf_counter()
            elapsed_ms = (end - start) * 1000
            self.logger.debug(f"process_b iteration took : {elapsed_ms:.2f} ms")
            if elapsed_ms > 1000:  # have to be <= 1 sec
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

        self.logger.debug("shutting down process_b")

        for handler in self.logger.handlers:
            handler.close()
