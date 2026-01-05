# standard library imports
import time
from datetime import datetime
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
from candidate_saver import CandidateSaver
from beam_check_config import (
    SAMPLES_PER_SECOND,
    BUFFER_LENGTH,
    BPM_NAMES,
    NANOSECS_IN_1_SEC,
    CANDIDATE_LOOKBACK_WINDOW_LENGTH,
    ANOMALY_CANDIDATE_WINDOW_SIZE,
)

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
        self.buffer = Buffer(
            pv_list=self.pv_list,
            buffer_len=BUFFER_LENGTH,
            snapshot_length=SAMPLES_PER_SECOND,
            snapshot_period_ns=NANOSECS_IN_1_SEC,
            logging_kwargs=self.logging_kwargs.copy(),
        )  # 36000 = 120hz * 60sec/min * 5mins
        # holds anomaly candidates
        self.candidate_bucket = CandidateBucket()
        self.candidate_saver = CandidateSaver('saved_candidates', self.logging_kwargs.copy())
        self.last_anomaly_timestamps = {k: -1 for k in self.pv_list if k.endswith("PHAS_FASTBR")}

    def __call__(self) -> None:
        if self.logger is None:
            self.logger = create_worker_logger(**self.logging_kwargs)

        self.logger.info(f"Starting process_b for {len(self.pv_list)} PVs")
        self.logger.debug("Beginning main data processing loop...")

        while True:
            # need to be sure each iteration of this processing loop is <= 1 second
            # (data comes each second from process_a, so data will pile-up if our processing takes over 1 second)
            timer_start = time.perf_counter()
            data_in_snapshot = True
            try:
                snapshot = self.queue_one.get(timeout=0.05)  # wait 50ms
            except Empty:
                self.logger.debug("No new data in queue_one (timeout reached).")
            else:
                if snapshot is None:  # enqueuing a None should stop this process immediately
                    self.logger.info("Received shutdown signal. Stopping process")
                    self.queue_two.put(None)
                    break
                self.logger.debug("Received new snapshot from queue_one")

                # parse the k2eg snapshot and update buffer
                index_change, length_of_update = self.buffer.update(snapshot)
                if index_change == 0 and length_of_update == 0:
                    data_in_snapshot = False
                    self.logger.warning(f"No data in snapshot {snapshot['iteration']:d} (skipping processing)")
                else:
                    self.logger.debug(
                        f"Buffer updated: index_change={index_change}, length_of_update={length_of_update}"
                    )
                    self.look_for_new_candidates(index_change, length_of_update)
            finally:
                if data_in_snapshot:
                    self.look_for_ready_candidates()

            elapsed_ms = (time.perf_counter() - timer_start) * 1000
            self.logger.debug(f"process_b loop iteration took {elapsed_ms:.2f} ms")
            if elapsed_ms > 1000:  # have to be <= 1 sec
                self.logger.warning(f"process_b is slow! Iteration took {elapsed_ms:.2f} ms")

        self.logger.info("Shutting down process_b")

        for handler in self.logger.handlers:
            handler.close()

    def look_for_new_candidates(self, index_change: int, length_of_update: int) -> None:
        self.logger.debug("Looking for new anomaly candidates...")
        # move the indexes of the previously found candidates
        self.candidate_bucket.update_slow_indexes(index_change)
        # add new candidates to the bucket
        new_candidates: List[AnomalyCandidate] = self.buffer.find_candidates(look_back_this_far=length_of_update)

        for candidate in new_candidates:
            self.candidate_bucket.put(candidate)

    def look_for_ready_candidates(self) -> None:
        # right now buffer.find_candidates checks every time point in the window and converts them all into candidates
        # Jason's analysis applies two filters: (1) filters on 'beam health' and (2) silences candidates for a little
        # while after a candidate is produced.  Implement (1) and (2) here
        ss = f"Bucket has {len(self.candidate_bucket):d} candidates, "
        ss += "checking for ready candidates..."
        self.logger.debug(ss)
        acws = ANOMALY_CANDIDATE_WINDOW_SIZE
        # check for candidates ready for process C
        while self.candidate_bucket.oldest_candidate_slow_index <= self.buffer.index - acws:
            self.logger.debug("Getting the oldest candidate for processing")
            candidate = self.candidate_bucket.get()  # get the oldest candidate

            cand = self.process_candidate(candidate=candidate)

            if cand:  # dictionary is not empty for some reason
                evaluation = self.evaluate_candidate(cand)
                eval_cand = cand | {"reject_reason": evaluation}
                if evaluation.lower() == 'none':
                    self.candidate_saver.save_anomaly_candidate(eval_cand)
                    self.queue_two.put(cand)
                    log_verb = "detected"
                    log_method = self.logger.info
                else:  # reject the candidate
                    self.candidate_saver.save_anomaly_candidate(eval_cand, reject=True)
                    log_verb = "rejected"
                    log_method = self.logger.debug
                fast_time = cand["candidate_timestamp"]
                ts = str(datetime.fromtimestamp(fast_time / NANOSECS_IN_1_SEC))
                ss = (f"Anomaly candidate {log_verb:s} at time: {ts}, PV: {cand['rf_pv_name']},"
                      f" Score: {cand['anomaly_score']:.2f}")
                log_method(ss)

    def process_candidate(self, candidate: AnomalyCandidate) -> dict:
        self.logger.debug("Starting to process candidate...")
        score_start_index = max(0, candidate.slow_index - CANDIDATE_LOOKBACK_WINDOW_LENGTH)
        self.logger.debug(f"Candidate: slow_index={candidate.slow_index}, score_start_index={score_start_index}")

        # find the most anomalous rf station
        station_names = [n for n in self.pv_list if n.endswith("PHAS_FASTBR")]
        rf_phase_data = np.stack(
            [self.buffer.get(pv, score_start_index, candidate.slow_index) for pv in station_names], axis=1
        )  # (CANDIDATE_WINDOW_SIZE, len(station_names))
        most_anomalous_rf_pv_name, deviation_score, system_level_anom = find_most_anomalous_rf_station(
            rf_phase_data, rf_pv_names=station_names
        )
        self.logger.debug(f"Most anomalous rf PV: {most_anomalous_rf_pv_name}, Score: {deviation_score:.2f}")

        # Get the bpm_score_20 values for the lookback window
        bpm_score_20 = self.buffer.get("bpm_score_20", score_start_index, candidate.slow_index)

        # find the fast trigger
        fast_index = score_start_index + find_fast_index(bpm_score_20)
        candidate.fast_index = fast_index
        fast_time = self.buffer.get("pv_timestamps_ns", fast_index, fast_index + 1)[0]
        self.logger.debug(f"Fast trigger index: {fast_index}, timestamp: {fast_time}")

        # prepare anomaly candidate data for process C
        # TODO: integrate beam check results into slow/fast index selection
        anomaly_data_window = candidate.window_slice
        data_quality_array = self.buffer.get("beam_checks", anomaly_data_window[0], anomaly_data_window[1]).copy()
        rf_input: np.array = self.buffer.get(
            most_anomalous_rf_pv_name, anomaly_data_window[0], anomaly_data_window[1]
        ).copy()
        bpm_input = []
        for pv_name in BPM_NAMES:
            bpm_input.append(self.buffer.get(pv_name, anomaly_data_window[0], anomaly_data_window[1]).copy())
        # send candidate to process C
        if most_anomalous_rf_pv_name:  # non-empty rf_pv_name
            self.logger.debug(f"Returning anomaly candidate dict for PV '{most_anomalous_rf_pv_name}'")
            return {
                "candidate_timestamp": fast_time,
                "rf_input": rf_input.reshape(1, -1),
                "bpm_input": np.vstack(bpm_input),
                "rf_pv_name": most_anomalous_rf_pv_name,
                "anomaly_score": deviation_score,
                "system_level_anomaly": system_level_anom,
                "number_of_bad_datapoints": sum(data_quality_array),
                "data_quality_array": data_quality_array,
            }
        else:
            self.logger.debug("No valid RF PV name found, returning empty candidate")
            return {}

    def evaluate_candidate(self, candidate: dict) -> str:
        """
        Returns a string describing why the candidate should be rejected.  None means keep the candidate.

        Parameters
        ----------
        candidate : dict
            A dictionary from self.process_candidate.

        Returns
        -------
        A string describing why the candidate should be rejected.
        """
        if not candidate:  # if the candidate is empty
            return "no rf station found responsible"
        if not np.all(candidate["data_quality_array"]):
            return "not all candidate data is healthy"
        
        last_anomaly_timestamp = self.last_anomaly_timestamps[candidate["rf_pv_name"]]
        current_candidate_timestamp = candidate["candidate_timestamp"].item()
        wait_time_sec = 1
        if current_candidate_timestamp - last_anomaly_timestamp < wait_time_sec * NANOSECS_IN_1_SEC:
            return "too soon since last candidate"

        # if there is no reason to reject the candidate, keep it and update last_anomaly_timestamp
        self.last_anomaly_timestamps[candidate["rf_pv_name"]] = current_candidate_timestamp
        return "none"

