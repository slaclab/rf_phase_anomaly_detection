import numpy as np

from run_config import BEAM_RATE_PV, BEAM_SPLIT_PV, STOPPER_PV, IN_TMIT_PV, EXP_TMIT_MIN


BEAM_CHECK_PVS = [BEAM_RATE_PV, BEAM_SPLIT_PV, STOPPER_PV, IN_TMIT_PV]


def do_beam_checks(check_signals: dict[str, np.ndarray]) -> np.ndarray:
    """
    Does some basic checks to see if the beam is healthy enough to look
    for anomalies.

    The checks use cryptic codes, they mean:
    The beam rate must be 120 Hz (BEAM_RATE_PV == 8)
    The beam split must be to HXR only (BEAM_SPLIT_PV == 10)
    The beam stopper must be out (STOPPER_PV == 0)
    The beam charge must be more than 0.5e9 (IN_TMIT_PV > EXP_TMIT_MIN)

    You can find dictionaries for these PVs in the config file the PV
    names are imported from.

    Parameters
    ----------
    check_signals: dict[str, np.ndarray]
        A dictionary of PVs to check.

    Returns
    -------
        numpy array of True and False; True means the beam is healthy
    enough to use.
    """
    for pv_name in BEAM_CHECK_PVS:
        try:
            check_signals[pv_name]
        except KeyError:
            raise KeyError(f"The PV named {pv_name:s} must be passed to do_beam_checks")

    data_is_good = np.vstack(
        (
            check_signals[BEAM_RATE_PV] == 8,  # ensure the beam rate is 120 Hz
            check_signals[BEAM_SPLIT_PV] == 10,  # ensure the beam is going only to HXR
            check_signals[STOPPER_PV] == 0,  # ensure the stopper is out
            check_signals[IN_TMIT_PV] > EXP_TMIT_MIN,  # ensure the charge reading is large enough
        )
    )

    return np.logical_and.reduce(data_is_good, axis=0)
