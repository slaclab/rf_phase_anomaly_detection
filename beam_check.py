from datetime import datetime, timedelta
from typing import Dict
import numpy as np
import pandas as pd
from buffer import Buffer

# Constants
BEAM_RATE_PV = "IOC:BSY0:MP01:PC_RATE"
BEAM_SPLIT_PV = "IOC:IN20:EV01:RG02_ACTRATE"
IN_TMIT_PV = "BPMS:IN20:221:TMITCUHBR" #?? do we want "BR" at end of this pv-name
STOPPER_PV = "STPR:BSYH:2:STD2_IN_A"

BEAM_RATE_TABLE = {1: 0, 4: 1, 5: 10, 6: 30, 7: 60, 8: 120}
BEAM_SPLIT_TABLE_HXR = {
    1: 0, 2: 0, 3: 1, 4: 10, 5: 30, 6: 60,
    7: 90, 8: 110, 9: 119, 10: 120, 11: 1,
    12: 10, 13: 0, 14: 0
}

MIN_VIOLATION_DUR = timedelta(seconds=90)
EXP_TMIT_FREQ = 1
ALLOWED_TMIT_DIFF = 0.05
EXP_TMIT_MIN = 0.5e9

def do_beam_checks(buffer: Buffer, starting_index: int, num_samples_to_check: int) -> None:
    """
    Apply beam checks to data in buffer_map, from starting_index to starting_index+num_samples_to_check.
    Fills buffer.passes_beam_checks with boolean results.
    """
    for curr_index in range(starting_index, num_samples_to_check):

        # Stopper check
        stopper = buffer.buffer_map.get("ca://"+STOPPER_PV)[curr_index]
        stopper_clear = stopper == 0

        # Beam rate check (must be 120Hz)
        beam_rate = buffer.buffer_map.get("ca://"+BEAM_RATE_PV)[curr_index]
        full_rate = BEAM_RATE_TABLE.get(beam_rate, 0) == 120

        # Beam split check (must be 120Hz HXR)
        beam_split = buffer.buffer_map.get("ca://"+BEAM_SPLIT_PV)[curr_index]
        hxr_split = BEAM_SPLIT_TABLE_HXR.get(beam_split, 0) == 120

        # TMIT check (must be real charge + logged at ~1Hz)
        tmit = buffer.buffer_map.get("ca://"+IN_TMIT_PV)[curr_index]
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