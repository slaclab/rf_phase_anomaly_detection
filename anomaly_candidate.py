from queue import PriorityQueue
import numpy as np

from beam_check_config import (ANOMALY_CANDIDATE_WINDOW_SIZE, FEEDBACK_STATIONS, RF_PV_NAMES, CANDIDATE_PHASE_THRESHOLD)


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


def find_fast_index(bpm_score_20: np.ndarray) -> int:
    """
        Implements the  fast trigger from Section IV A of
        https://arxiv.org/abs/2505.16052

        Computes a simple heuristic to determine the earliest time point within the sequence where the anomaly occurs)
        https://github.com/SLAC-ML/CoincAD/blob/phase/core/h5_dataloader_bpm_trigger_2024_TriggerMulti_8ch_centerPhasTrigger_asym_cleanup_final.ipynb
        Function - get_fast_trigger_idx

        Parameters
        ----------
        bpm_score_20: np.ndarray
            bpm-score values for the lookback window

        Returns
        -------
            Index in bpm_score_20 where fast trigger is detected
    """
    if len(bpm_score_20) < 10:  # fallback if data is too small
        return len(bpm_score_20)

    # Split into two parts: baseline (first part of the window) and trigger_window (last_half)
    baseline_end = max(1, len(bpm_score_20) // 2)
    baseline = bpm_score_20[:baseline_end]
    trigger_window = bpm_score_20[baseline_end:]

    # Compute mean and std from baseline window
    baseline_mean = np.mean(baseline)
    baseline_std = np.std(baseline) + 1e-6  # avoid divide-by-zero

    # Compute z-scores in trigger window
    z_scores = np.abs(trigger_window - baseline_mean) / baseline_std

    # Find first point above threshold (z > 1.25)
    indexes_above_thresh = np.where(z_scores > 1.25)[0]

    if len(indexes_above_thresh) > 0:  # Found anomaly; get first one
        rel_fast_idx = baseline_end + indexes_above_thresh[0]
    else:
        rel_fast_idx = (
            baseline_end + len(trigger_window) // 2
        )  # No clear anomaly; default to center of trigger_window

    # Return fast trigger index in absolute buffer coordinates
    return rel_fast_idx


def find_most_anomalous_rf_station(
    rf_phase_data: np.ndarray,
    rf_pv_names: list[str] = RF_PV_NAMES,
    phas_thresh: float = CANDIDATE_PHASE_THRESHOLD,
) -> tuple[str, float, bool]:
    """
    Implements the RF Candidate Selection from Appendix C of
    https://arxiv.org/abs/2505.16052

    Identifies the most anomalous klystron station identification based on phase deviation
    https://github.com/SLAC-ML/CoincAD/blob/phase/core/h5_dataloader_bpm_trigger_2024_TriggerMulti_8ch_centerPhasTrigger_asym_cleanup_final.ipynb
    Function - most_anomalous_klys

    Parameters
    ----------
    rf_phase_data : np.ndarray
        2D array containing rf station data.
    rf_pv_names : list[str]
        List of RF PV names corresponding to phas_fast channels.
    phas_thresh : float
        Threshold to flag a system-level phase scan.

    Returns
    -------
    Tuple[str, float, bool]
        The most anomalous RF PV name, its deviation score, and a system_level flag (0 or 1).
    """

    # Absolute deviation from 0 (centered signal)
    rf_phase_data_abs = np.abs(rf_phase_data)

    # Max deviation per RF PV in this window
    max_per_rf = np.nanmax(rf_phase_data_abs, axis=0)  # shape: (num_rf_pvs,)

    # System-level anomaly check: more than 10 RFs over threshold
    system_level_anomaly = np.sum(max_per_rf > phas_thresh) > 10

    # Rank the top 5 highest deviations
    top5_indices = np.argsort(max_per_rf)[::-1]

    # Return the top *non-feedback* station
    # will fail if all rf stations are FEEDBACK_STATIONS
    for i in top5_indices:
        pv = rf_pv_names[i]
        if pv not in FEEDBACK_STATIONS:
            return pv, max_per_rf[i], system_level_anomaly

    return '', 0., system_level_anomaly


if __name__ == "__main__":
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
    window = np.stack([np.random.randn(window_size) for _ in RF_PV_NAMES], axis=1)  # (window_size, num_rf_pvs)
    most_anomalous_rf_pv_name, deviation_score, system_level_flag = find_most_anomalous_rf_station(
        window,
        rf_pv_names=RF_PV_NAMES
    )
    print(most_anomalous_rf_pv_name, deviation_score, system_level_flag)

    bpm_score_20 = np.random.rand(120)
    start = 0
    fast_index = find_fast_index(
        bpm_score_20=bpm_score_20,
        start_index=start
    )
    print(fast_index)
