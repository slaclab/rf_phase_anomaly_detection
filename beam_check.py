from datetime import datetime, timedelta
from typing import Dict
import numpy as np
import pandas as pd

# Constants
BEAM_RATE_PV = "IOC:BSY0:MP01:PC_RATE"
BEAM_SPLIT_PV = "IOC:IN20:EV01:RG02_ACTRATE"
IN_TMIT_PV = "BPMS:IN20:221:TMITCUH"
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

# Assumes snapshot is a dict of {PV_name: pd.Series}, aligned in time
def get_beam_checks_from_snapshot(snapshot: Dict[str, pd.Series]) -> pd.DataFrame:
    # 1. Stopper check (False if stopper is in)
    stopper = snapshot[STOPPER_PV] == 0
    stopper.name = "stopper_clear"

    # 2. Beam rate check (must be 120Hz)
    rate_values = snapshot[BEAM_RATE_PV].map(lambda v: BEAM_RATE_TABLE.get(v, 0))
    full_rate = rate_values == 120
    full_rate.name = "full_rate"

    # 3. Beam split check (must be 120Hz HXR)
    split_values = snapshot[BEAM_SPLIT_PV].map(lambda v: BEAM_SPLIT_TABLE_HXR.get(v, 0))
    hxr_split = split_values == 120
    hxr_split.name = "hxr_split"

    # 4. TMIT check (must be real charge + logged at ~1Hz)
    tmit = snapshot[IN_TMIT_PV]
    is_real_charge = tmit > EXP_TMIT_MIN
    is_real_charge.name = "real_charge"

    # Sampling interval check (naive diff method)
    tmit_time = tmit.index
    dt = tmit_time.to_series().diff().dt.total_seconds()
    is_logged_correctly = (dt.fillna(EXP_TMIT_FREQ).abs() - EXP_TMIT_FREQ).abs() <= ALLOWED_TMIT_DIFF
    is_logged_correctly.name = "logged_correctly"

    # Combine TMIT validity
    tmit_valid = is_real_charge & is_logged_correctly
    tmit_valid.name = "tmit_ok"

    # Combine all checks into one DataFrame
    return pd.concat([stopper, full_rate, hxr_split, tmit_valid], axis=1)

# Example usage
# snapshot = {
#     STOPPER_PV: pd.Series(...),
#     BEAM_RATE_PV: pd.Series(...),
#     BEAM_SPLIT_PV: pd.Series(...),
#     IN_TMIT_PV: pd.Series(...)
# }
# checks_df = get_beam_checks_from_snapshot(snapshot)
# checks_df['beam_ok'] = checks_df.all(axis=1)

