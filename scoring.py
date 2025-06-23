import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from beam_check_config import MAD_LENGTH, CONSECUTIVE_LENGTH


def compute_score_1(
        bpm_signals: dict[str, np.ndarray]
) -> np.ndarray:
    """
    Implements the first anomaly score from Section IV A of
    https://arxiv.org/abs/2505.16052

    Computes median-absolute deviation with a 600 point rolling window
    and then aggregates across all channels with a geometric mean.

    Parameters
    ----------
    bpm_signals : dict[str, np.ndarray]
        A dictionary

    Returns
    -------
        np.ndarray of the MAD for the last part of the input arrays
    """
    individual_scores = []
    for pv_name, time_series in bpm_signals.items():
        median = np.median(
            sliding_window_view(time_series, window_shape=MAD_LENGTH),
            axis=-1
        )
        # make median the same length as input data
        median = np.hstack((np.zeros(MAD_LENGTH - 1), median))
        deviation = np.abs(time_series - median)
        mad = np.median(
            sliding_window_view(deviation, window_shape=MAD_LENGTH),
            axis=-1
        )
        clipped_mad = np.clip(mad, a_min=1e-3, a_max=None)
        individual_scores.append(
            deviation[-mad.shape[0]:] / (1.4826 * clipped_mad)
        )
    scores = np.array(individual_scores)
    # do geometric mean across the BPMs
    return np.power(np.prod(scores, axis=0), 1/8)


def compute_score_20(
        bpm_scores: np.ndarray
) -> np.ndarray:
    """
    Implements the second anomaly score from Section IV A of
    https://arxiv.org/abs/2505.16052

    Computes the geometric mean of a 20 point rolling window.
    Parameters
    ----------
    bpm_scores : np.ndarray
        An array of data over which to compute the rolling
        geometric mean.

    Returns
    -------
        nd.array of the rolling geometric mean
    """
    windows = sliding_window_view(bpm_scores, window_shape=CONSECUTIVE_LENGTH)
    return np.power(np.prod(windows, axis=-1), 1/CONSECUTIVE_LENGTH)
