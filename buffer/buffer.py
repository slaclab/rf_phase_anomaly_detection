import os
import numpy as np

from buffer.beam_check import do_beam_checks, BEAM_CHECK_PVS
from run_config import MAD_LENGTH, BPM_NAMES, BPM_THRESHOLD, SAMPLES_PER_SECOND, NANOSECS_IN_1_SEC
from buffer.scoring import compute_score_1, compute_score_20
from buffer.sliding_window import SlidingWindowArray
from anomaly_candidate import AnomalyCandidate
from mp_logging import create_worker_logger, default_logging_kwargs
from buffer.snapshot_fixer import SnapshotFixer, get_timestamp_ns, get_value

from typing import Optional, Tuple


def get_first_time_point(snapshot: dict[str, list[dict]], pv_name_list: list[str]) -> int:
    """
    Finds the earliest time point in the snapshot.
    """
    # reference_ts = snapshot["timestamp"] / 1e3  # measured in ms
    min_ts = None
    for pv in pv_name_list:
        sn = snapshot.get(pv, [])
        if len(sn) > 0:
            ts = get_timestamp_ns(sn[0])
            # diff_ts = ts / 1e9 - reference_ts  # should be -1 at most
            # if (min_ts is None) or ((ts < min_ts) and (diff_ts >= -1)):
            if min_ts is None or ts < min_ts:
                min_ts = int(ts)
    return min_ts


def get_latest_time_point(snapshot: dict[str, list[dict]], pv_name_list: list[str]) -> int:
    """
    Finds the latest time point in the snapshot.
    """
    max_ts = None
    for pv in pv_name_list:
        sn = snapshot.get(pv, [])
        if len(sn) > 0:
            ts = get_timestamp_ns(sn[-1])
            if max_ts is None or ts > max_ts:
                max_ts = int(ts)
    return max_ts


def get_all_timestamps_from_pv(pv_list: list[dict]) -> list[int]:
    return [get_timestamp_ns(x) for x in pv_list]


class Buffer:
    """
    Fixed-length buffer for storing a sliding window of 120hz float data per pv.
    By default stores 5 mins (36000 values) of past data.
    """

    def __init__(
        self,
        pv_list: list[str],
        buffer_len: int,
        snapshot_length: int,
        snapshot_period_ns: int,
        logging_kwargs: Optional[dict] = default_logging_kwargs,
    ):
        logging_kwargs["logger_name"] = "buffer"
        self.logger = create_worker_logger(**logging_kwargs)

        self.pv_list = pv_list  # list of PVs to buffer
        self.buffer_len = buffer_len  # max length of buffer
        self.snapshot_period_ns = snapshot_period_ns  # length of a snapshot in nanoseconds
        self.snapshot_length = snapshot_length  # length of a snapshot in number of entries

        self.fixer = SnapshotFixer(pv_list=pv_list, logging_kwargs=logging_kwargs.copy())

        self.init_or_reinit_buffer(reinit=False, logging_kwargs=logging_kwargs)

    def update(self, snapshot: dict[str, list[dict]]) -> Tuple[int, int]:
        """
        Append the latest 120-sample PV snapshot into the buffer for each pv

        The snapshots are assumed to return a value for each PV, even if that value is very old.

        Return
        ------
            Two integers.  The first is the number of indexes data might have
            been moved back.  The second is the length of the snapshot.
        """
        self.logger.debug("Buffer starting update...")

        # if this is the first snapshot seen, get the oldest time across all pv data-points
        if self.time_of_first_data == -1:
            min_ts = get_latest_time_point(snapshot, self.pv_list)
            if min_ts is not None:
                # the first data is expected to show up one bucket after the last sample found:
                self.time_of_first_data = min_ts + NANOSECS_IN_1_SEC // SAMPLES_PER_SECOND
                ss = f"Snapshot {snapshot['iteration']}, start time for all data is {self.time_of_first_data:d} ns"
                self.logger.info(ss)
            else:
                ss = f"Buffer failed to find a starting time for data collection in snapshot {snapshot['iteration']}"
                self.logger.warning(ss)
            was_full_before_new_data = False
            snapshot_length = 0
        else:  # otherwise, process the data
            # get the expected time window (start and end) for this snapshot
            start_time = self.time_of_first_data + self.num_snapshots_processed * self.snapshot_period_ns
            end_time = start_time + self.snapshot_period_ns

            # makes sure that all new data is within the above time window, also gets the latest datapoint
            # that should have already been sent, if any
            fixed_snapshot_data, new_latest_values = self.fixer.fix_snapshot(snapshot, start_time, end_time)
            # copy any new latest values into memory
            for key, value in new_latest_values.items():
                self.prev_snapshot_val_map[key] = value
            # bucket the data into 120 Hz buckets
            fixed_and_bucketed_snapshot_data = self.fixer.bucket_snapshot_data(
                fixed_snapshot_data, self.prev_snapshot_val_map, start_time, end_time
            )

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
                    {name: self.data_map[name].get(self.index - total_length, self.index) for name in BPM_NAMES}
                )

                bpm_score_20 = compute_score_20(bpm_score_1)
                self.logger.debug("Computed bpm_score_20")

                self.data_map["bpm_score_1"].put(bpm_score_1[-self.snapshot_length:])
                self.data_map["bpm_score_20"].put(bpm_score_20[-self.snapshot_length:])
            else:  # append 0's to keep bpm_score arrays same length as pv arrays
                self.data_map["bpm_score_1"].put(np.zeros(self.snapshot_length))
                self.data_map["bpm_score_20"].put(np.zeros(self.snapshot_length))
                self.logger.debug("Not enough data yet for bpm score computation, adding zeros to bpm_score array")

            self.logger.debug(f"num snapshots processed {self.num_snapshots_processed}")
            self.num_snapshots_processed += 1

            self.logger.debug("Buffer done updating")
            snapshot_length = self.snapshot_length

        # update the latest PV value map with the last value from each pv
        # we assume here that (1) k2eg always returns a value, even if very old,
        # (2) the values returned are in chronological order
        failed_updates = []
        # self.num_snapshots_processed was increased by 1 a few lines above
        # so this is the start of the next snapshot window
        next_start_time = (self.time_of_first_data +
                           self.num_snapshots_processed * self.snapshot_period_ns)
        for pv in self.pv_list:
            prev_snapshot_val = None
            try:
                # # this can pull from the future, but it is quick:
                # prev_snapshot_val = get_value(snapshot[pv][-1], self.logger)
                # respects time of arrival, but does more work:
                for entry in snapshot[pv]:
                    if get_timestamp_ns(entry) <= next_start_time:
                        prev_snapshot_val = get_value(entry, self.logger)
                    else:  # entries are in time order, break when you are into the future
                        break
            except IndexError:  # this exception is caused by the -1 in snapshot[pv][-1]
                failed_updates.append((pv, 'empty'))
            except KeyError:  # this exception is caused by the pv in snapshot[pv]
                failed_updates.append((pv, 'dne'))
            else:
                if prev_snapshot_val is not None:
                    self.prev_snapshot_val_map[pv] = prev_snapshot_val
                elif self.prev_snapshot_val_map[pv] is not None:
                    # if you get here prev_snapshot_val is None but you have an older value stored, keep going
                    msg = f"PV {pv:s} for snapshot number {snapshot['iteration']:d} "
                    msg += "does not have any values this snapshot, using older values"
                    self.logger.debug(msg)
                else:
                    # if you get here, it is because snapshot[pv] has values out of order
                    # and the first value comes after next_start_time, or you have lost a snapshot somewhere
                    # since you do not already have self.prev_snapshot_val_map[pv] saved, you have to start over
                    msg = f"PV {pv:s} for snapshot number {snapshot['iteration']:d} "
                    msg += f"appears to be out of order or entirely from the future. Length: {len(snapshot[pv])}. "
                    msg += f"next_start_time: {next_start_time}"
                    self.logger.warning(msg)
                    wrong_time_stamps = get_all_timestamps_from_pv(snapshot[pv])
                    msg = f"PV timestamps: {wrong_time_stamps}"
                    self.logger.warning(msg)
                    msg = f"Time relative to next_start_time: {[(x - next_start_time) / 1e9 for x in wrong_time_stamps]} seconds"
                    self.logger.warning(msg)
                    return -1, -1  # tell ProcessB there was a fatal problem

        # # Dump the first many snapshots for diagnostics purposes
        # uu = snapshot['iteration']
        # if uu < 50:
        #     import json
        #     save_this = snapshot | {
        #         'next_start_time': next_start_time,
        #         'time_of_first_data': self.time_of_first_data,
        #         'num_snapshots_processed': self.num_snapshots_processed,
        #         'snapshot_period_ns': self.snapshot_period_ns
        #     }
        #     with open(f"snapshots/snapshot_{uu:03d}.json", 'w') as jf:
        #         json.dump(save_this, jf, indent=4)

        if len(failed_updates) > 0:
            ss = f"There were some issues updating Buffer.prev_snapshot_val_map "
            ss += f"for snapshot {snapshot['iteration']:d}: "
            for name in ['empty', 'dne']:
                c = sum([1 for x in failed_updates if x[-1] == name])
                ss += f"{c:d} were {name} | "
            self.logger.warning(ss[:-3])

        return -self.snapshot_length if was_full_before_new_data else 0, snapshot_length

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

    def init_or_reinit_buffer(
            self,
            reinit: Optional[bool] = False,
            logging_kwargs: Optional[dict] = default_logging_kwargs
    ) -> None:
        """
        This initializes or re-initializes the stateful parts of the buffer.
        """
        self.num_snapshots_processed = 0
        self.time_of_first_data = -1
        self.index = 0  # tracks the next write index
        self.prev_snapshot_val_map = {}  # store the latest value for each PV for potential forward-filling

        if not reinit:  # first initialization
            self.logger.debug(f"Initializing buffer for {len(self.pv_list)} PVs, buffer length = {self.buffer_len}")
            # default array length is 36000 to store 5 mins of data at 120hz.
            # we allocate the arrays initially to avoid potential memory-copies during array append operation.
            self.data_map = {
                pv: SlidingWindowArray(self.buffer_len, dtype=np.float64, pv_name=pv, logging_kwargs=logging_kwargs.copy())
                for pv in self.pv_list
            }
            # just store the timestamp data from the first pv we read from the snapshot,
            # and assume the other pv's data is timed the same.
            self.data_map["pv_timestamps_ns"] = SlidingWindowArray(
                self.buffer_len, dtype=np.int64, pv_name="pv_timestamps_ns", logging_kwargs=logging_kwargs.copy()
            )
            self.data_map["beam_checks"] = SlidingWindowArray(
                self.buffer_len, dtype=bool, pv_name="beam_checks", logging_kwargs=logging_kwargs.copy()
            )
            self.data_map["bpm_score_1"] = SlidingWindowArray(
                self.buffer_len, dtype=np.float64, pv_name="bpm_score_1", logging_kwargs=logging_kwargs.copy()
            )
            self.data_map["bpm_score_20"] = SlidingWindowArray(
                self.buffer_len, dtype=np.float64, pv_name="bpm_score_20", logging_kwargs=logging_kwargs.copy()
            )
            # just normal arr for valid_windows, since doesn't have a max size and need sliding logic to drop old values
            # self.data_map["valid_windows"] = set()  # will hold tuples of (window_start_index, window_end_index)
        else:  # re-initialize the SlidingWindowArrays
            self.logger.debug(f"Reinitializing buffer for {len(self.pv_list)} PVs, buffer length = {self.buffer_len}")
            for sliding_window_array in self.data_map.values():
                sliding_window_array.init_sliding_window_array()


if __name__ == "__main__":
    from machine_interface.k2eg_spoofer import K2EGSpoofer  # adjust import as needed
    from utilities import read_pv_list_from_file

    list_of_pvs = read_pv_list_from_file("../resources/pv_list.txt")
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
