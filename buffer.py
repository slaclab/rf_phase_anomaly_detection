from typing import Dict, Optional, Tuple
import numpy as np
import os

from beam_check import do_beam_checks, BEAM_CHECK_PVS
from beam_check_config import MAD_LENGTH, BPM_NAMES, SAMPLES_PER_SECOND, BPM_THRESHOLD
from scoring import compute_score_1, compute_score_20
from sliding_window import SlidingWindowArray
from anomaly_candidate import AnomalyCandidate
from mp_logging import create_worker_logger, default_logging_kwargs
from snapshot_fixer import SnapshotFixer, get_timestamp_ns


class Buffer:
    """
    Fixed-length buffer for storing a sliding window of 120hz float data per pv.
    By default stores 5 mins (36000 values) of past data.
    """

    def __init__(self, pv_list: list[str], buffer_len: int, snapshot_length: int, snapshot_period_ns: int, logging_kwargs: Optional[dict] = default_logging_kwargs):

        logging_kwargs["logger_name"] = "buffer"
        self.logger = create_worker_logger(**logging_kwargs)

        self.pv_list = pv_list

        self.num_snapshots_processed = 0
        self.time_of_first_data = -1

        # max length of buffer
        self.buffer_len = buffer_len

        self.index = 0  # tracks the next write index

        self.logger.debug(f"Initializing buffer for {len(pv_list)} PVs, buffer length = {buffer_len}")

        # default array length is 36000 to store 5 mins of data at 120hz.
        # we allocate the arrays initially to avoid potential memory-copies during array append operation.
        self.data_map = {pv: SlidingWindowArray(buffer_len, dtype=np.float64, pv_name=pv, logging_kwargs=logging_kwargs.copy()) for pv in self.pv_list}
        # just store the timestamp data from the first pv we read from the snapshot,
        # and assume the other pv's data is timed the same.
        self.data_map["pv_timestamps_ns"] = SlidingWindowArray(buffer_len, dtype=np.int64, pv_name="pv_timestamps_ns", logging_kwargs=logging_kwargs.copy())
        self.data_map["beam_checks"] = SlidingWindowArray(buffer_len, dtype=bool, pv_name="beam_checks", logging_kwargs=logging_kwargs.copy())
        self.data_map["bpm_score_1"] = SlidingWindowArray(buffer_len, dtype=np.float64, pv_name="bpm_score_1", logging_kwargs=logging_kwargs.copy())
        self.data_map["bpm_score_20"] = SlidingWindowArray(buffer_len, dtype=np.float64, pv_name="bpm_score_20", logging_kwargs=logging_kwargs.copy())
        # just normal arr for valid_windows, since doesn't have a max size and need sliding logic to drop old values
        self.data_map["valid_windows"] = set()  # will hold tuples of (window_start_index, window_end_index)

        self.snapshot_period_ns = snapshot_period_ns
        self.snapshot_length = snapshot_length

        self.fixer = SnapshotFixer(pv_list=pv_list, logging_kwargs=logging_kwargs.copy())

    def update(self, snapshot: dict[str, list[dict]]) -> Tuple[int, int]:
        """
        Append the latest 120-sample PV snapshot into the buffer for each pv

        Return
        ------
            Two integers.  The first is the number of indexes data might have
            been moved back.  The second is the length of the snapshot.
        """
        self.logger.debug("Buffer starting update...")

        # map of pv to last value from prev snapshot (or 0 if this is the first snapshot)
        # (used for potential forward-filling)
        prev_snapshot_val_map = {}
        for pv in self.pv_list:
            if self.num_snapshots_processed == 0:
                prev_snapshot_val_map[pv] = 0.0
            else:
                prev_snapshot_val_map[pv] = self.data_map[pv].get(-1)

        # get oldest time across all pv data-points
        if self.time_of_first_data == -1:
            min_ts = None
            for pv in self.pv_list:
                for e in snapshot.get(pv, []):
                    ts = get_timestamp_ns(e)
                    if min_ts is None or ts < min_ts:
                        min_ts = ts
            if min_ts is None:  # snapshot is empty (no data)
                return (0,0)
            else:
                self.time_of_first_data = min_ts

        start_time = self.time_of_first_data + self.num_snapshots_processed * self.snapshot_period_ns
        end_time = start_time + self.snapshot_period_ns

        fixed_snapshot_data = self.fixer.fix_snapshot(snapshot, start_time, end_time)
        fixed_and_bucketed_snapshot_data = self.fixer.bucket_snapshot_data(fixed_snapshot_data, prev_snapshot_val_map, start_time, end_time)

        beam_check_data = {}
        for pv in BEAM_CHECK_PVS:
            beam_check_data[pv] = fixed_and_bucketed_snapshot_data[pv]

        # now update our global map with bucket-data
        for pv, values in fixed_and_bucketed_snapshot_data.items():
            # self.logger.debug(f"Appending bucketed + cleaned data to data_map for PV: {pv}")
            self.data_map[pv].put(values)

        # do the beam checks and put the data on beam_check buffer
        beam_checks_result = do_beam_checks(beam_check_data)
        self.data_map["beam_checks"].put(beam_checks_result)

        self.index = self.data_map[self.pv_list[0]].index  # use first pv as index reference

        was_full_before_new_data = self.data_map["bpm_score_20"].is_full()

        total_length = self.snapshot_length + MAD_LENGTH - 1
        if self.index > total_length:
            bpm_score_1 = compute_score_1(
                {
                    name: self.data_map[name].get(
                        self.index - total_length, self.index
                    )
                    for name in BPM_NAMES
                }
            )

            bpm_score_20 = compute_score_20(bpm_score_1)
            self.logger.debug(f"Computed bpm_score_20")

            self.data_map["bpm_score_1"].put(bpm_score_1[-self.snapshot_length:])
            self.data_map["bpm_score_20"].put(bpm_score_20[-self.snapshot_length:])
        else: # append 0's to keep bpm_score arrays same length as pv arrays
            self.data_map["bpm_score_1"].put(np.zeros(self.snapshot_length))
            self.data_map["bpm_score_20"].put(np.zeros(self.snapshot_length))
            self.logger.debug("Not enough data yet for bpm score computation, adding zeros to bpm_score array")

        self.logger.debug(f"num snapshots processed {self.num_snapshots_processed}")
        self.num_snapshots_processed += 1

        self.logger.debug("Buffer done updating")
        return (-self.snapshot_length if was_full_before_new_data else 0, self.snapshot_length)

    def find_candidates(self, look_back_this_far: int) -> list[AnomalyCandidate]:
        candidates = []
        start = self.index - look_back_this_far
        end = self.index

        # avoid errors when trying to find candidates b4 bpm_score_20 can be calculated,
        # as in: self.index < self.snapshot_length + MAD_LENGTH - 1
        # TODO: figure out if this is correct way to handle this
        if len(self.data_map["bpm_score_20"]) == 0:
            self.logger.debug("Skipping candidate search, bpm_score_20 array is empty")
            return []
        scores = self.data_map["bpm_score_20"].get(start, end)
        timestamps_ns = self.data_map["pv_timestamps_ns"].get(start, end)

        for i, (score, slow_time_ns) in enumerate(zip(scores, timestamps_ns)):
            if score > BPM_THRESHOLD:
                candidate = AnomalyCandidate(slow_index=start + i, slow_time=slow_time_ns)
                self.logger.debug(f"Candidate detected: slow_index: {candidate.slow_index}, score: {score:.2f}")
                candidates.append(candidate)

        return candidates

    def get(self, pv_name: str, start_index: Optional[int] = None, end_index: Optional[int] = None) -> np.ndarray:
        """
        Get data from the buffer map for given pv.
        Will return all the valid data for specified pv in buffer if start_index and end_index are None,
        else will return the data in the specified range. (or an empty array if the specified range is not valid)
        """
        if pv_name not in self.data_map:
            self.logger.error(f"pv_name '{pv_name}' not found in buffer map")
            raise KeyError(f"pv_name '{pv_name}' not found in buffer map")

        s = start_index if start_index is not None else 0
        e = end_index if end_index is not None else self.index

        if s < 0 or s > self.index or e > self.index:
            self.logger.warning(f"start and end indices not valid in buffer: {s}, {e}")
            raise IndexError(f"start and end indices not valid in buffer: {s}, {e}")

        return self.data_map[pv_name].get(s, e)

    def clear(self) -> None:
        """
        Clear buffer contents for all PVs.
        """
        self.logger.info("Clearing all buffer contents")
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
            valid_data = self.data_map[pv][: self.index]
            filename = pv.lstrip("ca://").replace(":", "_") + ".txt"
            filepath = os.path.join(directory, filename)
            self.logger.debug(f"writing dump file {filepath} for {pv}")
            with open(filepath, "w") as f:
                for v in valid_data:
                    f.write(f"{v}\n")


if __name__ == "__main__":
    from k2eg_spoofer import K2EGSpoofer  # adjust import as needed
    from k2eg_process import read_pv_list_from_file

    list_of_pvs = read_pv_list_from_file("resources/pv_list.txt")
    BUFFER_LENGTH = 36000

    spoofer = K2EGSpoofer(
        pv_configs=[{"name": name, "rate_hz": 120, "drop_rate": 0.0} for name in list_of_pvs], n_emits=4, emit_rate_hz=1
    )
    spoofed_data = list(spoofer())

    # create buffer and fill it with spoofed data
    buffer = Buffer(list_of_pvs, BUFFER_LENGTH)

    for i, emission in enumerate(spoofed_data):
        index_change, length_of_update = buffer.update(emission)
        print(f"updated buffer #{i}: index_change={index_change}, length_of_update={length_of_update}")

        candidates = buffer.find_candidates(look_back_this_far=length_of_update)
        print(f"Found {len(candidates)} candidates")