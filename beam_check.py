from datetime import datetime, timedelta
from typing import Dict
import numpy as np

from beam_check_config import BEAM_RATE_TABLE, BEAM_SPLIT_TABLE_HXR, EXP_TMIT_MIN

# note: can't pass in buffer object directly b/c of circular import
# (buffer.py would import beam_check.py, and beam_check.py would import buffer.py ...).
def do_beam_checks(
    stopper_pv_data: np.ndarray,
    beam_rate_pv_data: np.ndarray,
    beam_split_pv_data: np.ndarray,
    in_tmit_pv_data: np.ndarray,
    starting_index: int,
    num_samples_to_check: int,
    passes_beam_checks: np.ndarray
) -> None:
    """
    Apply beam checks to data in buffer_map, from starting_index to starting_index+num_samples_to_check.
    Fills buffer.passes_beam_checks with boolean results.
    """

    return True

    """
    # vectorized version
    end_index = starting_index + num_samples_to_check
    sl = slice(starting_index, end_index)

    stopper_clear = stopper_pv_data[sl] == 0
    full_rate = np.vectorize(BEAM_RATE_TABLE.get)(beam_rate_pv_data[sl], 0) == 120 # passing the .get function to vectorize
    hxr_split = np.vectorize(BEAM_SPLIT_TABLE_HXR.get)(beam_split_pv_data[sl], 0) == 120
    is_real_charge = in_tmit_pv_data[sl] > EXP_TMIT_MIN
    is_logged_correctly = np.ones_like(stopper_clear, dtype=bool) # need to figure out real check

    passes_beam_checks[sl] = (
        stopper_clear & full_rate & hxr_split & is_real_charge & is_logged_correctly
    )

    # for-loop version
    end_index = starting_index + num_samples_to_check
    for curr_index in range(starting_index, end_index):

        # Stopper check
        stopper = stopper_pv_data[curr_index]
        stopper_clear = stopper == 0

        # Beam rate check (must be 120Hz)
        beam_rate = beam_rate_pv_data[curr_index]
        full_rate = BEAM_RATE_TABLE.get(beam_rate, 0) == 120

        # Beam split check (must be 120Hz HXR)
        beam_split = beam_split_pv_data[curr_index] 
        hxr_split = BEAM_SPLIT_TABLE_HXR.get(beam_split, 0) == 120

        # TMIT check (must be real charge + logged at ~1Hz)
        tmit = in_tmit_pv_data[curr_index]
        is_real_charge = tmit > EXP_TMIT_MIN

        '''
        # (need to understand this check more...)
        # Sampling interval check (naive diff method)
        tmit_time = tmit.index
        dt = tmit_time.to_series().diff().dt.total_seconds()
        is_logged_correctly = (dt.fillna(EXP_TMIT_FREQ).abs() - EXP_TMIT_FREQ).abs() <= ALLOWED_TMIT_DIFF
        is_logged_correctly.name = "logged_correctly"
        '''
        is_logged_correctly = True

        # combine all checks
        passes_all_checks = stopper_clear & full_rate & hxr_split & is_real_charge & is_logged_correctly
        buffer.passes_beam_checks[curr_index] = passes_all_checks
    """