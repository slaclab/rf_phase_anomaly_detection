"""
Test the TimedDict and TimedCountDict classes from anom_table.
"""

import pytest
import time
from datetime import datetime
from collections import deque

from anom_table import (TimedBoolDict, TimedCandCountDict, TimedAnomCountDict,
                        format_pv_name_for_table, load_klystron_configs)

KEY = "KLYS:LI20:61:PHAS_FASTCUHBR"
F_KEY = "klys_li20_61"
RESET_TIME = 0.1

####################################################################################
# fixtures
####################################################################################

@pytest.fixture(scope="module")
def create_timedbooldict() ->TimedBoolDict:
    return TimedBoolDict(
        write_to_pv=False,
        queue_inst=None,
        reset_time=RESET_TIME,
    )

@pytest.fixture(scope="module")
def create_timedcandcountdict() ->TimedCandCountDict:
    return TimedCandCountDict(
        write_to_pv=False,
        queue_inst=None,
        reset_time=RESET_TIME,
    )

@pytest.fixture(scope="module")
def create_timedanomcountdict() ->TimedAnomCountDict:
    return TimedAnomCountDict(
        write_to_pv=False,
        queue_inst=None,
        reset_time=RESET_TIME,
    )

####################################################################################
# functions
####################################################################################

def test_key_name_changer():
    f_key = format_pv_name_for_table(KEY)
    assert f_key == F_KEY

####################################################################################
# TimedBoolDict
####################################################################################

def test_create_timedbooldict(create_timedbooldict):
    tbd = create_timedbooldict
    assert tbd.default_value == False
    assert tbd.data == {format_pv_name_for_table(k): tbd.default_value for k in load_klystron_configs()}
    assert tbd.timers == {}
    assert tbd.write_to_pv_method == 'update_anomaly_state'

def test_tbd__reset_key(create_timedbooldict):
    tbd = create_timedbooldict
    tbd.data[F_KEY] = True
    tbd.timers[F_KEY] = None  # does not matter what this is
    tbd._reset_key(KEY)
    assert tbd.data[F_KEY] == False
    assert F_KEY not in tbd.timers

def test_tbd_get_dict(create_timedbooldict):
    tbd = create_timedbooldict
    tbd.data[F_KEY] = None
    dd = tbd.get_dict()
    assert dd[F_KEY] == None
    tbd._reset_key(KEY)

def test_tbd_set_key(create_timedbooldict):
    tbd = create_timedbooldict
    tbd.set_key(KEY, True)     # should set to True and create a timer
    assert tbd.data[F_KEY] == True   # checks True was set
    assert F_KEY in tbd.timers       # checks timer was created
    time.sleep(RESET_TIME + 0.2)     # waits for timer to run out
    assert tbd.data[F_KEY] == False  # checks True was replaced by False
    assert F_KEY not in tbd.timers   # checks that timer was destroyed

def test_tbd_shut_down(create_timedbooldict):
    tbd = create_timedbooldict
    tbd.set_key(KEY, True)  # should set to True and create a timer
    tbd.shut_down()
    assert tbd.timers == {}

####################################################################################
# TimedCandCountDict
####################################################################################

def test_create_timedcantcountdict(create_timedcandcountdict):
    tcd = create_timedcandcountdict
    assert tcd.default_value == None
    assert tcd.data == {format_pv_name_for_table(k): tcd.default_value for k in load_klystron_configs()}
    assert tcd.timers == {}
    assert tcd.write_to_pv_method == 'update_cand_count'

def test_tcd__reset_key(create_timedcandcountdict):
    tcd = create_timedcandcountdict
    tcd.data[F_KEY] = [45]  # does not matter what this is, cannot be None, must have length
    tcd._reset_key(KEY)
    assert tcd.data[F_KEY] == None

def test_tcd_get_dict(create_timedcandcountdict):
    tcd = create_timedcandcountdict
    dd = tcd.get_dict()
    assert all([v == 0 for v in dd.values()])

def test_tcd_drop_old(create_timedcandcountdict):
    tcd = create_timedcandcountdict
    ts = datetime.now().timestamp() - RESET_TIME - 10
    tcd.data[F_KEY] = deque([ts], maxlen=3000)
    assert isinstance(tcd.data[F_KEY], deque)
    assert len(tcd.data[F_KEY]) == 1
    tcd.drop_old()
    assert len(tcd.data[F_KEY]) == 0

def test_tcd_set_key(create_timedcandcountdict):
    tcd = create_timedcandcountdict
    ts = datetime.now().timestamp() + 0.5  # set the time in the future to keep RESET_TIME short
    tcd.set_key(KEY, ts)
    assert isinstance(tcd.data[F_KEY], deque)
    assert len(tcd.data[F_KEY]) == 1
    assert tcd.data[F_KEY][0] == ts
    time.sleep(1)  # wait long enough for the timer to call drop_old
    assert len(tcd.data[F_KEY]) == 0
    tcd.shut_down()

####################################################################################
# TimedAnomCountDict
# same as TimedCandCountDict but for the write_to_pv_method attribute
####################################################################################

def test_create_timedanomcountdict(create_timedanomcountdict):
    tad = create_timedanomcountdict
    assert tad.default_value == None
    assert tad.data == {format_pv_name_for_table(k): tad.default_value for k in load_klystron_configs()}
    assert tad.timers == {}
    assert tad.write_to_pv_method == 'update_anom_count'

