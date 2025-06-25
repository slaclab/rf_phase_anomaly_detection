from typing import Optional, Tuple
import numpy as np
import logging

from mp_logging import create_worker_logger
from beam_check import do_beam_checks
from beam_check_config import (
    BEAM_RATE_PV,
    BEAM_SPLIT_PV,
    IN_TMIT_PV,
    STOPPER_PV,
    BEAM_RATE_TABLE,
    BEAM_SPLIT_TABLE_HXR,
    EXP_TMIT_FREQ,
    ALLOWED_TMIT_DIFF,
    EXP_TMIT_MIN,
    MAD_LENGTH,
    CONSECUTIVE_LENGTH,
    BPM_NAMES,
    SAMPLES_PER_SECOND,
    BPM_THRESHOLD
)
from scoring import compute_score_1, compute_score_20
from sliding_window import SlidingWindowArray
from anomaly_candidate import AnomalyCandidate, CandidateBucket


class Buffer:
    """
    Fixed-length buffer for storing a sliding window of 120hz float data per pv.
    By default stores 5 mins (36000 values) of past data.
    """
    def __init__(self, pv_list: list[str], buffer_len: int, logger: logging.Logger) -> None:
        self.pv_list = pv_list

        # max length of buffer
        self.buffer_len = buffer_len

        self.index = 0 # tracks the next write index
        # default array length is 36000 to store 5 mins of data at 120hz.
        # we allocate the arrays initially to avoid potential memory-copies during array append operation.
        self.data_map = {
            pv: SlidingWindowArray(buffer_len, dtype=np.float64) for pv in self.pv_list
        }
        # just store the timestamp data from the first pv we read from the snapshot,
        # and assume the other pv's data is timed the same.
        self.data_map["pv_timestamps_ns"] = SlidingWindowArray(buffer_len, dtype=np.float64)
        self.data_map["beam_checks"] = SlidingWindowArray(buffer_len, dtype=bool)
        self.data_map["bpm_score_1"] = SlidingWindowArray(buffer_len, dtype=np.float64)
        self.data_map["bpm_score_20"] = SlidingWindowArray(buffer_len, dtype=np.float64)
        # just normal arr for valid_windows, since doesn't have a max size and need sliding logic to drop old values
        self.data_map["valid_windows"] = set() # will hold tuples of (window_start_index, window_end_index)

        self.logger = logger


    def update(self, snapshot: dict[str, list[dict]]) -> Tuple[int, int]:
        """
        Append the latest 120-sample PV snapshot into the buffer for each pv

        Return
        ------
            Two integers.  The first is the number of indexes data might have
            been moved back.  The second is the length of the snapshot.
        """
        snapshot_length = 120
        for i, pv in enumerate(self.pv_list):
            entries = snapshot.get(pv, [])

            values = np.empty(SAMPLES_PER_SECOND, dtype=np.float64)
            for j, e in enumerate(entries):
                values[j] = e.get("value", np.nan)

            self.data_map[pv].put(values)

            # update buffer's timestamps arr with the first pv's times,
            # and assume the other pv have same timing (for now).
            if i == 0:
                timestamps_ns = np.empty(SAMPLES_PER_SECOND, dtype=int)
                for j, e in enumerate(entries):
                    ts = e.get("timeStamp", {})
                    seconds = ts.get("secondsPastEpoch", 0)
                    nanos = ts.get("nanoseconds", 0)
                    timestamps_ns[j] = int(seconds * 1e9 + nanos)

                self.data_map["pv_timestamps_ns"].put(timestamps_ns)

        self.clean_data()

        beam_checks_result = do_beam_checks(
            stopper_pv_data=self.data_map[STOPPER_PV],
            beam_rate_pv_data=self.data_map[BEAM_RATE_PV],
            beam_split_pv_data=self.data_map[BEAM_SPLIT_PV],
            in_tmit_pv_data=self.data_map[IN_TMIT_PV],
            #min_violation_window_length_check=self.data_map["min_violation_window_length_check"],
            starting_index=self.index,
            num_samples_to_check=SAMPLES_PER_SECOND,
        )

        self.data_map["beam_checks"].put(beam_checks_result)

        self.index = self.data_map[self.pv_list[0]].index  # use first pv as index reference

        was_full_before_new_data = self.data_map["bpm_score_20"].is_full()

        total_length = SAMPLES_PER_SECOND + MAD_LENGTH - 1
        if self.index > total_length:
            bpm_score_1 = compute_score_1({
                name: self.data_map.get(name, self.index - total_length, self.index) for name in BPM_NAMES
            })
            bpm_score_20 = compute_score_20(bpm_score_1)
            self.data_map["bpm_score_1"].put(bpm_score_1[-SAMPLES_PER_SECOND:])
            self.data_map["bpm_score_20"].put(bpm_score_20[-SAMPLES_PER_SECOND:])
            #Candidate Gen
            self.bpm_candidate_bucket.update_slow_indexes(-SAMPLES_PER_SECOND)

        return (
            -snapshot_length if was_full_before_new_data else 0,
            snapshot_length
        )


    def clean_data(self) -> None:
        # forward fill data not updated during timestamp, check if corresponding pv timestamps are close enough, etc
        return


    def find_candidates(self, look_back_this_far: int) -> list[AnomalyCandidate]:
        candidates = []
        start = self.index - look_back_this_far
        end = self.index
        scores = self.data_map["bpm_score_20"].get(start, end)
        timestamps_ns = self.data_map["pv_timestamps_ns"].get(start, end)

        for i, (score, slow_time_ns) in enumerate(zip(scores, timestamps_ns)):
            if score > BPM_THRESHOLD:
                candidate = AnomalyCandidate(slow_index=start + i, slow_time=slow_time_ns)
                candidates.append(candidate)

        return candidates


    def get(self, pv_name: str, start_index: Optional[int] = None, end_index: Optional[int] = None) -> np.ndarray:
        """
        Get data from the buffer map for given pv.
        Will return all the valid data for specified pv in buffer if start_index and end_index are None,
        else will return the data in the specified range. (or an empty array if the specified range is not valid)
        """
        if pv_name not in self.data_map:
            raise KeyError(f"pv_name '{pv_name}' not found in buffer map")

        if start_index < 0 or start_index > self.index or end_index > self.index:
            raise IndexError(f"start and end indicies not valid in buffer: {start_index}, {end_index}")
            return []

        s = start_index if start_index is not None else 0
        e = end_index if end_index is not None else self.index
        return self.data_map[pv_name][s:e]

    def clear(self) -> None:
        """
        Clear buffer contents for all PVs.
        """
        for pv_name in self.data_map:
            self.data_map[pv_name].clear()
        self.data_map["pv_timestamps_ns"].clear()
        self.data_map["beam_checks"].clear()

    def dump_to_human_readable(self, directory: str = "buffer_dump_txt") -> None:
        """
        Debug util: dump each PV's data to a separate text file, 1 value per line.

        Call like this:
            dir_name = f"buffer_txt_dump_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.logger.debug(f"dump dir: {dir_name}")
            self.buffer.dump_to_human_readable(directory=dir_name)
        """
        os.makedirs(directory, exist_ok=True)

        for pv in self.pv_list:
            valid_data = self.data_map[pv][:self.index]
            filename = pv.lstrip("ca://").replace(':', '_') + ".txt"
            filepath = os.path.join(directory, filename)
            self.logger.debug(f"writing dump file {filepath} for {pv}")
            with open(filepath, "w") as f:
                for v in valid_data:
                    f.write(f"{v}\n")
