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

    This function partially replicates the function 
    bpm_extractor.scorer.score from
    https://github.com/slaclab/SLAC-AD-REF/blob/trig-bpm-fixed/papers/phase/BPM_Source_final.py#L156C12-L156C105

    First, the dispersions are used to scale some of the BPM readings:
    https://github.com/slaclab/SLAC-AD-REF/blob/trig-bpm-fixed/papers/phase/energy_anomaly_8ch.py#L124
    Second, MAD is computed for each channel
    https://github.com/slaclab/SLAC-AD-REF/blob/trig-bpm-fixed/papers/phase/energy_anomaly_8ch.py#L143
    Third, the geometric mean is taken across all 8 channels
    https://github.com/slaclab/SLAC-AD-REF/blob/trig-bpm-fixed/papers/phase/energy_anomaly_8ch.py#L150


    Parameters
    ----------
    bpm_signals : dict[str, np.ndarray]
        A dictionary

    Returns
    -------
        np.ndarray of the MAD for the entire array
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
        clipped_mad = np.hstack((
            1e-3 * np.ones(MAD_LENGTH - 1),
            np.clip(mad, a_min=1e-3, a_max=None)
        ))
        individual_scores.append(
            deviation / (1.4826 * clipped_mad)
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
    https://github.com/slaclab/SLAC-AD-REF/blob/trig-bpm-fixed/papers/phase/energy_anomaly_8ch.py#L155

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
