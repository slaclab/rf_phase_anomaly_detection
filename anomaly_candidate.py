from queue import PriorityQueue
import numpy as np

from run_config import ANOMALY_CANDIDATE_WINDOW_SIZE, FEEDBACK_STATIONS, CANDIDATE_PHASE_THRESHOLD


class AnomalyCandidate:
    def __init__(self, slow_index: int, slow_time: int):
        """
        Anomaly candidate for use with CandidateBucket below and the
        buffer.Buffer.

        Parameters
        ----------
        slow_index : int
            The 'slow' time index for when the anomaly occurs.
        slow_time : int
            The corresponding slow time for when the anomaly occurs
            in nanoseconds since the epoch.
        """
        self._slow_index = slow_index
        self.slow_time = slow_time

        self.window = [-ANOMALY_CANDIDATE_WINDOW_SIZE, ANOMALY_CANDIDATE_WINDOW_SIZE]

        self._fast_index = None
        self.score = None
        # more things here?

    def __str__(self):
        ss = "AnomalyCandidate"
        ss += f"(slow_index={self.slow_index}"
        ss += f", slow_time={self.slow_time})"
        return ss

    def __repr__(self):
        return self.__str__()

    # the following are used in the priority queue
    def __eq__(self, other):
        return self.slow_index == other.slow_index

    def __lt__(self, other):
        return self.slow_index < other.slow_index

    def __gt__(self, other):
        return self.slow_index > other.slow_index

    # the above are used in the priority queue

    @property
    def slow_index(self):
        return self._slow_index

    @property
    def fast_index(self):
        return self._fast_index

    @fast_index.setter
    def fast_index(self, new_index: int):
        self._fast_index = new_index

    @slow_index.setter
    def slow_index(self, new_index: int):
        self._slow_index = new_index

    @property
    def window_slice(self) -> list[int]:
        return [self._fast_index + x for x in self.window]

    def __iadd__(self, index_change: int):
        self.slow_index += index_change

    def __isub__(self, index_change: int):
        self.slow_index -= index_change


class CandidateBucket(PriorityQueue):
    def __len__(self):
        # PriorityQueues just store a list in self.queue
        return len(self.queue)

    @property
    def oldest_candidate_slow_index(self) -> int:
        if not self.empty():
            cand = self.get()
            index = cand.slow_index
            self.put(cand)
        else:
            index = int(1e20)  # huge number
        return index

    def update_slow_indexes(self, index_change: int):
        candidates = []
        # pull the candidates off and change their indexes
        while not self.empty():
            cand = self.get()
            cand.slow_index += index_change
            candidates.append(cand)
        # put the candidates back on
        for candidate in candidates:
            self.put(candidate)


def find_fast_index(bpm_score_1: np.ndarray) -> int:
    """
    Implements the  fast trigger from Section IV A of
    https://arxiv.org/abs/2505.16052

    Computes a simple heuristic to determine the earliest time point within the sequence where the anomaly occurs)
    https://github.com/SLAC-ML/CoincAD/blob/phase/core/h5_dataloader_bpm_trigger_2024_TriggerMulti_8ch_centerPhasTrigger_asym_cleanup_final.ipynb
    Function - get_fast_trigger_idx

    Parameters
    ----------
    bpm_score_1: np.ndarray
        bpm-score values for the lookback window

    Returns
    -------
        Index in bpm_score_1 where fast trigger is detected
    """
    if len(bpm_score_1) < 10:  # fallback if data is too small
        return len(bpm_score_1)

    # Split into two parts: baseline (first part of the window) and the full trigger_window
    baseline_end = max(1, len(bpm_score_1) // 2)
    baseline = bpm_score_1[:baseline_end]
    trigger_window = bpm_score_1

    # Compute mean and std from baseline window
    baseline_mean = np.mean(baseline)
    baseline_std = np.std(baseline) + 1e-6  # avoid divide-by-zero

    # Compute absolute z-scores in trigger window
    z_scores = np.abs(trigger_window - baseline_mean) / baseline_std

    # Find first point above threshold (z > 1.25)
    indexes_above_thresh = np.where(z_scores > 1.25)[0]

    if len(indexes_above_thresh) > 0:  # Found anomaly; get first one
        rel_fast_idx = indexes_above_thresh[0]
    else:
        rel_fast_idx = len(trigger_window) // 2  # No clear anomaly; default to center of trigger_window

    # Return fast trigger index counting from the start of bpm_score_1
    return rel_fast_idx


def find_most_anomalous_rf_station(
    rf_phase_data: np.ndarray,
    rf_pv_names: list[str],
    phas_thresh: float = CANDIDATE_PHASE_THRESHOLD,
) -> list[tuple[str, float, bool]]:
    """
    Implements the RF Candidate Selection from Appendix C of
    https://arxiv.org/abs/2505.16052

    Identifies the most anomalous klystron station identification based on phase deviation
    https://github.com/SLAC-ML/CoincAD/blob/phase/core/h5_dataloader_bpm_trigger_2024_TriggerMulti_8ch_centerPhasTrigger_asym_cleanup_final.ipynb
    Function - most_anomalous_klys

    Parameters
    ----------
    rf_phase_data : np.ndarray
        2D array containing rf station data.  shape is [window_size, num_rf_pvs]
    rf_pv_names : list[str]
        List of RF PV names corresponding to phas_fast channels.
    phas_thresh : float
        Threshold to flag a system-level phase scan.

    Returns
    -------
    List[Tuple[str, float, bool]]
        List of the most anomalous stations:
        (RF PV name, its deviation score, and a system_level flag (0 or 1)) for each station.
        Returns an empty list if no non-feedback stations are above threshold.
    """

    # Absolute deviation from 0 (centered signal)
    # replaces NaN with zeros, so they get sorted to smallest
    rf_phase_data_abs = np.nan_to_num(np.abs(rf_phase_data), nan=0.)  # shape is [window_size, num_rf_pvs]

    # sort the values for each time point (row) along the rf-station axis (columns)
    # this sorts smallest to largest
    rf_phase_data_abs_sorted = np.sort(rf_phase_data_abs, axis=1)  # shape is [window_size, num_rf_pvs]

    # take the 5th largest value for each time point and tile for subtraction
    fifth = min(5, rf_phase_data_abs.shape[1])  # in case you get too few rf_stations
    phase_fifth_max = np.sort(rf_phase_data_abs_sorted, axis=1)[:, -fifth]  # shape is [window_size]
    phase_fifth_max = np.tile(
        phase_fifth_max.reshape(-1, 1),
        (1, rf_phase_data_abs.shape[-1])
    )  # shape is [window_size, num_rf_pvs]

    # Max deviation per RF PV in this window
    max_per_rf = np.nanmax(rf_phase_data_abs - phase_fifth_max, axis=0)  # shape: (num_rf_pvs,)

    # System-level anomaly check: more than 10 RFs over threshold
    system_level_anomaly = np.sum(max_per_rf > phas_thresh) > 10

    # Rank the top 5 highest deviations - sorts largest to smallest
    # use stable to always get the same ordering in case of ties
    top_indices = np.argsort(max_per_rf, stable=True)[::-1]

    # Return the top *non-feedback* station
    top_stations = []
    # will return [] if all rf stations are FEEDBACK_STATIONS
    for index in top_indices:
        max_val = max_per_rf[index]
        if max_val < phas_thresh and len(top_stations) > 0:
            # all further stations are lower, stop and return only those stations above threshold
            # return at least one station if none are above threshold
            break
        pv = rf_pv_names[index]
        if pv not in FEEDBACK_STATIONS:
            top_stations.append(
                (pv, max_val, system_level_anomaly)
            )
        if len(top_stations) >= 5:  # reached 5 stations, stop and return all 5
            break

    return top_stations


if __name__ == "__main__":
    from utilities import read_pv_list_from_file

    rf_pv_names = [n for n in read_pv_list_from_file("resources/pv_list.txt") if n.endswith("PHAS_FASTCUHBR")]

    bucket = CandidateBucket()

    # this will still work even though the bucket is empty
    bucket.update_slow_indexes(22)

    for i in range(5):
        bucket.put(AnomalyCandidate(i, i))

    print(bucket.queue)
    print(bucket.oldest_candidate_slow_index)
    item_0 = bucket.get()
    item_1 = bucket.get()
    print(bucket.oldest_candidate_slow_index)
    print(bucket.queue)
    bucket.put(item_0)
    print(bucket.queue)
    bucket.update_slow_indexes(-4)
    print(bucket.queue)

    candidate = bucket.get()
    window_size = 20
    window = np.stack([np.random.randn(window_size) for _ in rf_pv_names], axis=1)  # (window_size, num_rf_pvs)
    station_candidates = find_most_anomalous_rf_station(
        window, rf_pv_names=rf_pv_names
    )  # this will be empty
    # for sc in station_candidates:
    #     print(sc)

    bpm_score_1 = np.random.rand(120)
    start = 0
    fast_index = find_fast_index(bpm_score_1=bpm_score_1)
    print(fast_index)
