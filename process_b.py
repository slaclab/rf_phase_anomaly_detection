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
from beam_check_config import (SAMPLES_PER_SECOND, BUFFER_LENGTH, BPM_NAMES,
                               RF_PV_NAMES, CANDIDATE_LOOKBACK_WINDOW_LENGTH,
                               ANOMALY_CANDIDATE_WINDOW_SIZE)

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
        self.buffer = Buffer(pv_list, BUFFER_LENGTH, logging_kwargs.copy())  # 3600 = 120hz * 60sec * 5mins
        # holds anomaly candidates
        self.candidate_bucket = CandidateBucket()

    def __call__(self) -> None:
        if self.logger is None:
            self.logger = create_worker_logger(**self.logging_kwargs)

        self.logger.info(f"running process_b on {len(self.pv_list)} pvs")
        self.logger.info("starting data processing loop...")

        while True:
            # need to be sure each iteration of this processing loop is <= 1 second
            # (data comes each second from process_a, so data will pile-up if our processing takes over 1 second)
            timer_start = time.perf_counter()

            try:
                r = self.queue_one.get(timeout=0.05)  # wait 50ms
            except Empty:
                self.logger.debug("No new data in queue_one (timeout reached).")
                pass
            else:
                if r is None:  # enqueuing a None should stop this process immediately
                    self.logger.info("Received shutdown signal. Stopping process")
                    self.queue_two.put(None)
                    break

                # parse the k2eg snapshot and update buffer
                index_change, length_of_update = self.buffer.update(r)
                self.look_for_new_candidates(index_change, length_of_update)
            finally:
                self.look_for_ready_candidates()

            elapsed_ms = (time.perf_counter() - timer_start) * 1000
            self.logger.debug(f"process_b iteration took : {elapsed_ms:.2f} ms")
            if elapsed_ms > 1000:  # have to be <= 1 sec
                self.logger.warning(f"process_b buffer append is slow!! : {elapsed_ms:.2f} ms")

        self.logger.debug("shutting down process_b")

        for handler in self.logger.handlers:
            handler.close()

    def look_for_new_candidates(self, index_change: int, length_of_update: int) -> None:
        self.logger.debug("Looking for new anomaly candidates.")
        # move the indexes of the previously found candidates
        self.candidate_bucket.update_slow_indexes(index_change)
        # add new candidates to the bucket
        try:
            new_candidates: List[AnomalyCandidate] = self.buffer.find_candidates(
            look_back_this_far=length_of_update
            )
            self.logger.info(f"Found {len(new_candidates)} new candidates")
            for candidate in new_candidates:
                self.candidate_bucket.put(candidate)
                self.logger.debug(f"Candidate at slow_index={candidate.slow_index} added to bucket.")
        except Exception as e:
            self.logger.error(f"Error while searching for new candidates: {e}", exc_info=True)

    def look_for_ready_candidates(self) -> None:
        acws = ANOMALY_CANDIDATE_WINDOW_SIZE
        self.logger.debug("Checking for ready candidates to forward...")
        # check for candidates ready for process C
        while self.candidate_bucket.oldest_candidate_slow_index <= self.buffer.index - acws:
            try:
                candidate = self.candidate_bucket.get()  # get the oldest candidate
                self.logger.debug(f"Processing candidate at slow_index={candidate.slow_index}")
                cand = self.process_candidate(candidate=candidate)
                if cand:  # dictionary is not empty
                    self.queue_two.put(cand)
                    fast_time = cand["anomaly_timestamp"]
                    self.logger.info(f"Anomaly with timestamp {fast_time} sent to process C")
                else:
                    self.logger.warning("Empty candidate data returned. Skipping.")
            except Exception as e:
                self.logger.error("Failed to process or queue candidate: {e}", exc_info=True)

    def process_candidate(self, candidate: AnomalyCandidate) -> dict:
        score_start_index = max(0, candidate.slow_index - CANDIDATE_LOOKBACK_WINDOW_LENGTH)

        # find the most anomalous rf station
        rf_phase_data = np.stack(
            [self.buffer.get(pv, score_start_index, candidate.slow_index) for pv in RF_PV_NAMES], axis=1
            )  # (CANDIDATE_WINDOW_SIZE, len(RF_PV_NAMES)))

        try:
            most_anomalous_rf_pv_name, deviation_score, system_level_anom = find_most_anomalous_rf_station(
                rf_phase_data
            )
            self.logger.info(
                f"Most anomalous RF PV: {most_anomalous_rf_pv_name}, "
                f"score={deviation_score:.4f}, system_anom={system_level_anom}"
            )
        except Exception as e:
            self.logger.error("Error in identifying most anomalous RF station.", exc_info=True)
            return{}

        # Get the bpm_score_20 values for the lookback window
        bpm_score_20 = self.buffer.get("bpm_score_20", score_start_index, candidate.slow_index)

        # find the fast trigger
        fast_index = score_start_index + find_fast_index(bpm_score_20)
        candidate.fast_index = fast_index
        fast_time = self.buffer.get("pv_timestamp_ns", fast_index, fast_index + 1)[0]
        self.logger.info(f"Fast index: {fast_index}, timestamp: {fast_time}")

        # prepare anomaly candidate data for process C
        anomaly_data_window = candidate.window_slice
        data_quality_array = self.buffer.get(
            "beam_checks", anomaly_data_window[0], anomaly_data_window[1]
        ).copy()
        rf_input: np.array = self.buffer.get(
            most_anomalous_rf_pv_name, anomaly_data_window[0], anomaly_data_window[1]
        ).copy()
        bpm_input = []
        for pv_name in BPM_NAMES:
            bpm_input.append(
                self.buffer.get(pv_name, anomaly_data_window[0], anomaly_data_window[1]).copy()
            )
        # send candidate to process C
        if most_anomalous_rf_pv_name:  # non-empty rf_pv_name
            return {
                "anomaly_timestamp": fast_time,
                "rf_input": rf_input,
                "bpm_input": np.vstack(bpm_input),
                "rf_pv_name": most_anomalous_rf_pv_name,
                "anomaly_score": deviation_score,
                "system_level_anomaly": system_level_anom,
                "number_of_bad_datapoints": sum(data_quality_array)
            }
        else:
            return {}
