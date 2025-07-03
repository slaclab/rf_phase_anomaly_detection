from queue import PriorityQueue
import numpy as np

from beam_check_config import ANOMALY_CANDIDATE_WINDOW_SIZE, FEEDBACK_STATIONS, RF_PV_NAMES


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
            index = -1
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


def find_fast_index(
    buffer, slow_index: int, window_size: int, samples_per_second: int, lookback_seconds: int = 5
) -> int:
    """
    Implements the  fast trigger from Section IV A of
    https://arxiv.org/abs/2505.16052

    Computes a simple heuristic to determine the earliest time point within the sequence where the anomaly occurs)
    https://github.com/SLAC-ML/CoincAD/blob/phase/core/h5_dataloader_bpm_trigger_2024_TriggerMulti_8ch_centerPhasTrigger_asym_cleanup_final.ipynb
    Function - get_fast_trigger_idx

    Parameters
    ----------
    buffer: Buffer
        The rolling buffer object that stores phase PVs.
    slow_index: int
        Index of the slow trigger in the buffer.
    window_size: int
        Number of samples to include before the slow_index.
    samples_per_second: int
        Sampling rate of data (typically 120)
    lookback_seconds: int
        How far back from the slow index to search (default 5s)

    Returns
    -------
        Absolute index in buffer where fast trigger was detected
    """
    # Calculate number of samples to look back from the slow trigger index
    lookback = lookback_seconds * samples_per_second

    buffer_index = buffer.index
    start = max(0, slow_index - lookback)
    end = min(buffer_index, slow_index)

    # Get the bpm_score_1 values for the lookback window
    bpm_score_1 = buffer.get("bpm_score_1", start, end)

    # Compute the relative slow index
    rel_slow_idx = slow_index - start

    # Define the region to calculate baseline and detect change
    window_start = max(0, rel_slow_idx - window_size)
    window_end = rel_slow_idx
    window = bpm_score_1[window_start:window_end]

    if len(window) < 10:
        return slow_index  # fallback if data is too small

    # Split into two parts: baseline (first part of the window) and trigger_window (last_half)
    baseline_end = max(1, len(window) - window_size // 2)
    baseline = window[:baseline_end]
    trigger_window = window[baseline_end:]

    # Compute mean and std from baseline window
    baseline_mean = np.mean(baseline)
    baseline_std = np.std(baseline) + 1e-6  # avoid divide-by-zero

    # Compute z-scores in trigger window
    z_scores = np.abs(trigger_window - baseline_mean) / baseline_std

    # Find first point above threshold (z > 1.25)
    above_thresh = np.where(z_scores > 1.25)[0]

    if len(above_thresh) > 0:
        rel_fast_idx = window_start + baseline_end + above_thresh[0]  # Found anomaly; get first one
    else:
        rel_fast_idx = (
            window_start + baseline_end + len(trigger_window) // 2
        )  # No clear anomaly; default to center of trigger_window

    # Return fast trigger index in absolute buffer coordinates
    return start + rel_fast_idx


def find_most_anomalous_rf_station(
    window,
    rf_pv_names: list[str],
    phas_thresh: float = 2.5,
) -> tuple[str, float, int]:
    """
    Implements the RF Candidate Selection from Appendix C of
    https://arxiv.org/abs/2505.16052

    Identifies the most anomalous klystron station identification based on phase deviation
    https://github.com/SLAC-ML/CoincAD/blob/phase/core/h5_dataloader_bpm_trigger_2024_TriggerMulti_8ch_centerPhasTrigger_asym_cleanup_final.ipynb
    Function - most_anomalous_klys

    Parameters
    ----------
    window : np.ndarray
        2D array containing rf station data.
    rf_pv_names : list[str]
        List of RF PV names corresponding to phas_fast channels.
    phas_thresh : float
        Threshold to flag a system-level phase scan.

    Returns
    -------
    Tuple[str, float, int]
        The most anomalous RF PV name, its deviation score, and a system_level flag (0 or 1).
    """

    # Absolute deviation from 0 (centered signal)
    window_abs = np.abs(window)

    # Max deviation per RF PV in this window
    max_per_rf = np.nanmax(window_abs, axis=0)  # shape: (num_rf_pvs,)

    # System-level anomaly check: more than 10 RFs over threshold
    system_level = int(np.sum(max_per_rf > phas_thresh) > 10)

    # Rank the top 5 highest deviations
    top5_indices = np.argsort(max_per_rf)[-5:][::-1]

    # Return the top *non-feedback* station
    for i in top5_indices:
        pv = rf_pv_names[i]
        if pv not in FEEDBACK_STATIONS:
            return pv, max_per_rf[i], system_level

    # If all top stations are feedback stations, return top anyway
    return rf_pv_names[top5_indices[0]], max_per_rf[top5_indices[0]], system_level


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
        rf_pv_names=RF_PV_NAMES,
        phas_thresh=2.5,
    )
    print(most_anomalous_rf_pv_name, deviation_score, system_level_flag)
