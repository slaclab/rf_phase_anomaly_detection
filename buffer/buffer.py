import os
import numpy as np

from buffer.beam_check import do_beam_checks, BEAM_CHECK_PVS
from run_config import MAD_LENGTH, BPM_NAMES, BPM_THRESHOLD, SAMPLES_PER_SECOND, NANOSECS_IN_1_SEC
from buffer.scoring import compute_score_1, compute_score_20
from buffer.sliding_window import SlidingWindowArray
from anomaly_candidate import AnomalyCandidate
from mp_logging import create_worker_logger, default_logging_kwargs
from buffer.snapshot_fixer import SnapshotFixer, get_timestamp_ns, get_value
from data_export_functions import save_fixed_snapshot, save_bucketed_snapshot, save_sliding_windows

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

        # the following will immediately be replaced in init_or_reinit_buffer
        self.num_snapshots_processed = None
        self.time_of_first_data = None
        self.index = None  # tracks the next write index
        self.prev_snapshot_val_map = None
        self.init_mode = None  # True means init_mode, False means run_mode
        self.init_or_reinit_buffer(reinit=False, logging_kwargs=logging_kwargs)

    def update(self, snapshot: dict[str, list[dict]]) -> Tuple[int, int]:
        """
        Append the latest 120-sample PV snapshot into the buffer for each pv

        The snapshots are assumed to return a value for each PV, even if that value is very old.

        Return
        ------
            Two integers.  The first is the number of indexes data might have
            been moved back.  The second is the length of the snapshot.
            Output meanings:
            -1, -1    : fatal error, you will have to reinit the buffer
            0, 0      : fine but the buffer is not changing, init_mode hasn't completed
            0, 120    : buffer is full and is updating
            -120, 120 : buffer is not full and is updating
            Note that 120 is the standard LCLS operating frequency.  0, Y and -Y, Y are fine
            for any value of Y that is smaller than the buffer.
        """

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

        iteration: int = snapshot["iteration"]

        self.logger.debug("Buffer starting update...")
        # in order to run successfully, this buffer needs two things:
        # (1) a starting time for all the data
        # (2) a value in prev_snapshot_val_map to fill in data
        # 1 is satisfied when self.time_of_first_data > -1
        # 2 is satisfied when there are zero None values in self.prev_snapshot_val
        # when you have not yet started collecting data, get the oldest time across all pv data-points
        if self.init_mode:  # if in init mode
            # passing the start time as the largest possible int means that new_latest_values is always
            # filled with the last value, if any.  end_time does not matter
            _, new_latest_values = self.fixer.fix_snapshot(
                raw_snapshot=snapshot, start_time=np.iinfo(np.int64).max, end_time=0
            )
            # update self.prev_snapshot_val_map
            val_map_is_full = self.update_prev_snapshot_val_map(new_latest_values, snapshot['iteration'])
            if val_map_is_full:
                # if you are here, prev_snapshot_val_map is full, update start time
                min_ts = get_latest_time_point(snapshot, self.pv_list)
                if min_ts is not None:
                    # the first data is expected to show up one bucket after the last sample found:
                    self.time_of_first_data = min_ts + NANOSECS_IN_1_SEC // SAMPLES_PER_SECOND
                    ss = f"Snapshot {iteration}, start time for all data is {self.time_of_first_data:d} ns"
                    self.logger.info(ss)
                    self.init_mode = False  # can leave init_mode now
                else:
                    ss = f"Buffer failed to find a starting time for data collection in snapshot {snapshot['iteration']}"
                    self.logger.warning(ss)
            else:
                # if you are here, you still have not collected enough prev_snapshot_val_map values
                # do not do anything, the final return will be 0, 0 and process B will continue
                number_of_prev_nones = sum([1 for v in self.prev_snapshot_val_map.values() if v is None])
                ss = f"buffer.prev_snapshot_val_map is not yet complete on snapshot {iteration}"
                ss += f" ({number_of_prev_nones:>3d}/{len(self.prev_snapshot_val_map):>3d}) PVs still None"
                self.logger.info(ss)
            # # the following are always correct in init_mode
            # was_full_before_new_data = False
            # snapshot_length = 0
            return 0, 0
        else:  # run mode, process the data
            # get the expected time window (start and end) for this snapshot
            start_time = self.time_of_first_data + self.num_snapshots_processed * self.snapshot_period_ns
            end_time = start_time + self.snapshot_period_ns

            # makes sure that all new data is within the above time window, also gets the latest datapoint
            # that should have already been sent, if any
            fixed_snapshot_data, new_latest_values = self.fixer.fix_snapshot(snapshot, start_time, end_time)
            # TODO: comment this out
            save_fixed_snapshot(fixed_snapshot_data, new_latest_values, iteration)
            # bucket the data into 120 Hz buckets
            fixed_and_bucketed_snapshot_data = self.fixer.bucket_snapshot_data(
                fixed_snapshot_data, self.prev_snapshot_val_map, start_time, end_time
            )
            # TODO: comment this out
            save_bucketed_snapshot(fixed_and_bucketed_snapshot_data, iteration)

            # update self.prev_snapshot_val_map after running bucket_snapshot_data
            self.update_prev_snapshot_val_map(new_latest_values, snapshot['iteration'])

            beam_check_data = {}
            for pv in BEAM_CHECK_PVS:
                beam_check_data[pv] = fixed_and_bucketed_snapshot_data[pv]

            # now update our global map with bucket-data
            for pv, values in fixed_and_bucketed_snapshot_data.items():
                # self.logger.debug(f"Appending bucketed + cleaned data to data_map for PV: {pv}")
                self.data_map[pv].put(values)

            # TODO: comment this out
            # save the BPM data for debugging
            windows_to_save = {pv: self.data_map[pv].data for pv in BPM_NAMES}
            save_sliding_windows(windows_to_save, iteration)

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

            self.num_snapshots_processed += 1
            self.logger.debug(f"num snapshots processed {self.num_snapshots_processed}")

            self.logger.debug("Buffer done updating")
            snapshot_length = self.snapshot_length

            # this must be run after self.num_snapshot_processed += 1
            # to properly account for what the next timestamp is
            # healthy_prev_snapshot_val_map = self.update_prev_snapshot_val_map(snapshot)
            if self.prev_snapshot_val_map_is_full:
                return -self.snapshot_length if was_full_before_new_data else 0, snapshot_length
            else:  # any unhealthy prev_snapshot_val_map condition requires a re-init
                return -1, -1

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
        self.prev_snapshot_val_map = {k: None for k in self.pv_list}  # store the latest value for each PV for potential forward-filling
        self.init_mode = True  # True means init_mode, False means run_mode

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

    def update_prev_snapshot_val_map(
            self,
            update_dict: dict[str, float],
            iteration: Optional[int] = -1
    ) -> bool:
        """
        Updates self.prev_snapshot_val_map according to update_dict.
        update_dict might be empty.  It only contains values that have something to update.
        This method assumes that prev_snapshot_val_map[key] is None if that key has never been updated.
        update_dict is expected to come from snapshot_fixer.fix_snapshot

        Parameters
        ----------
        update_dict : dict[str, float]
            Dictionary of PV names (keys) to update.
        iteration : int
            Iteration number of the snapshot.

        Returns
        -------
            bool True if prev_snapshot_val_map[key] is full (no Nones) and False otherwise.
        """

        for pv in self.pv_list:
            try:
                self.prev_snapshot_val_map[pv] = update_dict[pv]
            except KeyError:  # key was not updated this snapshot
                if self.prev_snapshot_val_map[pv] is None:
                    msg = f"PV {pv:s} for snapshot number {iteration:d} "
                    msg += "does not yet have a prev_snapshot_val_map entry"
                    self.logger.warning(msg)


        if self.prev_snapshot_val_map_is_full:
            return True
        else:
            return False

    @property
    def prev_snapshot_val_map_is_full(self) -> bool:
        if sum([1 for v in self.prev_snapshot_val_map.values() if v is None]) == 0:
            return True
        return False

    # def update_prev_snapshot_val_map(
    #         self,
    #         snapshot: dict[str, list[dict]]
    # ) -> bool:
    #     """
    #     Update prev_snapshot_val_map.  This map is required to be full (no None values) for a
    #     successful run.  You cannot leave init_mode until this is the case.
    #
    #     Parameters
    #     ----------
    #     snapshot : k2eg snapshot
    #
    #     Returns
    #     -------
    #         bool True if prev_snapshot_val_map is full and healthy, False otherwise
    #     """
    #     # update the latest PV value map with the last value from each pv
    #     # this will be used the next time something is added to the buffer to fill
    #     # in any buckets that occur before the first value in the snapshot
    #     # we assume here that (1) k2eg always returns a value, even if very old,
    #     # (2) the values returned are in chronological order
    #     failed_updates = []
    #
    #     if self.init_mode:
    #         next_start_time = np.inf  # in init mode, take the last value every time
    #     else:  # run mode
    #         # during normal run time (not init_mode) self.num_snapshots_processed
    #         # was increased by 1 at the end of processing, so this is the start of the next
    #         # snapshot window
    #         next_start_time = (self.time_of_first_data +
    #                            self.num_snapshots_processed * self.snapshot_period_ns)
    #     for pv in self.pv_list:
    #         prev_snapshot_val = None
    #         try:
    #             # # this can pull from the future, but it is quick:
    #             # prev_snapshot_val = get_value(snapshot[pv][-1], self.logger)
    #             # respects time of arrival, but does more work:
    #             for entry in snapshot[pv]:
    #                 if get_timestamp_ns(entry) <= next_start_time:
    #                     prev_snapshot_val = get_value(entry, self.logger)
    #                 else:  # entries are in time order, break when you are into the future
    #                     break
    #         except IndexError:  # this exception is caused by the -1 in snapshot[pv][-1]
    #             failed_updates.append((pv, 'empty'))
    #         except KeyError:  # this exception is caused by the pv in snapshot[pv]
    #             failed_updates.append((pv, 'dne'))
    #         else:
    #             if prev_snapshot_val is not None:
    #                 # if you found a new latest value in the try clause, save it here for the next iteration
    #                 self.prev_snapshot_val_map[pv] = prev_snapshot_val
    #             elif self.prev_snapshot_val_map[pv] is not None:
    #                 # if you get here prev_snapshot_val is None but you have an older value stored, keep going
    #                 msg = f"PV {pv:s} for snapshot number {snapshot['iteration']:d} "
    #                 msg += "does not have any values this snapshot, using older values"
    #                 self.logger.debug(msg)
    #             else:
    #                 # if you get here, it is because snapshot[pv] has values out of order and the first value comes
    #                 # after next_start_time, or you have lost a snapshot (or entry) somewhere
    #                 if self.init_mode:
    #                     # data collection hasn't started yet, it is ok to have missing values
    #                     msg = f"PV {pv:s} for snapshot number {snapshot['iteration']:d} "
    #                     msg += "does not yet have a prev_snapshot_val_map entry"
    #                     self.logger.warning(msg)
    #                 else:
    #                     # since you do not already have self.prev_snapshot_val_map[pv] saved,
    #                     # and data collection has begun, you have to start over
    #                     msg = f"PV {pv:s} for snapshot number {snapshot['iteration']:d} "
    #                     msg += f"appears to be out of order or entirely from the future. Length: {len(snapshot[pv])}. "
    #                     msg += f"next_start_time: {next_start_time}"
    #                     self.logger.warning(msg)
    #                     wrong_time_stamps = get_all_timestamps_from_pv(snapshot[pv])
    #                     msg = f"PV timestamps: {wrong_time_stamps}"
    #                     self.logger.warning(msg)
    #                     msg = f"Time relative to next_start_time: {[(x - next_start_time) / 1e9 for x in wrong_time_stamps]} seconds"
    #                     self.logger.warning(msg)
    #
    #                     failed_updates.append((pv, 'ooo'))  # ooo = "out of order"
    #                     # # if data collection has started, and you cannot find or fill in data, you must exit
    #                     # return -1, -1  # tell ProcessB there was a fatal problem
    #
    #     if len(failed_updates) > 0:
    #         ss = f"There were some issues updating Buffer.prev_snapshot_val_map "
    #         ss += f"for snapshot {snapshot['iteration']:d}: "
    #         for name in ['empty', 'dne', 'ooo']:
    #             c = sum([1 for x in failed_updates if x[-1] == name])
    #             ss += f"{c:d} were {name} | "
    #         self.logger.warning(ss[:-3])
    #
    #     if self.init_mode or len(failed_updates) == 0:
    #         return True
    #     else:
    #         return False


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
