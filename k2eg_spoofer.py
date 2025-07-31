import time
import random
from copy import deepcopy

import numpy as np

from k2eg_process import read_pv_list_from_file
from beam_check_config import NUM_NANOSEC_IN_1_SEC

from typing import TypedDict, Optional


PVConfiguration = TypedDict(
    'PVConfiguration',
    {
        'name': str,        # the name of the PV
        'rate_hz': float,   # the reading rate in Hz
        'drop_rate': float  # how often a reading will be skipped, randomly
    }
)


full_emit = {
    'value': 30.960044860839844,
    'alarm': {'severity': 0, 'status': 0, 'message': ''},
    'timeStamp': {'secondsPastEpoch': 1749656693, 'nanoseconds': 701017948, 'userTag': 0},
    'display': {'limitLow': 0.0, 'limitHigh': 360.0, 'description': '', 'format': 'F9.3', 'units': 'deg'},
    'control': {'limitLow': 0.0, 'limitHigh': 360.0, 'minStep': 0.0},
    'valueAlarm': {'active': 0, 'lowAlarmLimit': 0.0, 'lowWarningLimit': 0.0, 'highWarningLimit': 0.0, 'highAlarmLimit': 0.0, 'lowAlarmSeverity': 0, 'lowWarningSeverity': 0, 'highWarningSeverity': 0, 'highAlarmSeverity': 0, 'hysteresis': 0}
}


def spoof_reading(timestamp: float) -> dict:
    """ timestamp is nanoseconds since the epoch """
    reading = deepcopy(full_emit)
    seconds = int(timestamp / NUM_NANOSEC_IN_1_SEC)
    reading['value'] = random.random()
    reading['timeStamp'] = {
        'secondsPastEpoch': seconds,
        'nanoseconds': int(timestamp - NUM_NANOSEC_IN_1_SEC * seconds),
        'userTag': 0
    }
    return reading


def spoof_constant_reading(timestamp: float, value: float) -> dict:
    """ timestamp is nanoseconds since the epoch """
    reading = deepcopy(full_emit)
    seconds = int(timestamp / NUM_NANOSEC_IN_1_SEC)
    reading['value'] = value
    reading['timeStamp'] = {
        'secondsPastEpoch': seconds,
        'nanoseconds': int(timestamp - NUM_NANOSEC_IN_1_SEC * seconds),
        'userTag': 0
    }
    return reading


class PVSpoofer:
    def __init__(self, config: PVConfiguration) -> None:
        self.config = config
        self.last_emit_time_ns = 0
        self.emission_period = 1 / self.config['rate_hz']

    def emit_readings(
            self,
            start_time: float,
            end_time: float,
            constant_reading: Optional[float] = None
    ) -> list[dict]:
        """
        start_time and end_time are both nanoseconds since the epoch
        """
        if end_time - self.last_emit_time_ns < self.emission_period * NUM_NANOSEC_IN_1_SEC:
            return []
        else:  # emit at least once
            n_emit = max(
                int(self.config['rate_hz'] * (end_time - start_time) / NUM_NANOSEC_IN_1_SEC),
                1
            )

        emit_list = []
        drop_count = random.randint(
            0,
            int(self.config['drop_rate'] * n_emit)
        )
        drop_these = sorted(random.sample(range(n_emit), drop_count))
        for i in range(n_emit):
            timestamp = start_time + i * (end_time - start_time) / n_emit
            if i not in drop_these:  # not optimal
                if constant_reading is None:
                    emit_list.append(spoof_reading(timestamp))
                else:
                    emit_list.append(spoof_constant_reading(timestamp, constant_reading))
                self.last_emit_time_ns = timestamp
        return emit_list


class K2EGSpoofer:
    def __init__(
            self,
            pv_configs: list[PVConfiguration],
            n_emits: Optional[int] = 0,
            emit_rate_hz: Optional[int] = 1
    ) -> None:
        self.pv_configs = pv_configs
        self.n_emits = n_emits
        self.emit_rate_hz = emit_rate_hz
        self.emit_period = 1 / self.emit_rate_hz

        self.pv_spoofers = {pvc['name']: PVSpoofer(pvc) for pvc in self.pv_configs}

    def __call__(self):
        iteration = 0
        if self.n_emits <= 0:
            i = int(-1e10)
        else:
            i = 0

        while i < self.n_emits:
            start_time = time.time_ns()
            # do some stuff
            pvs = {
                k: v.emit_readings(start_time, start_time + self.emit_period * NUM_NANOSEC_IN_1_SEC)
                for k, v in self.pv_spoofers.items()
            }
            emission = {
                'iteration': iteration,
                'timestamp': int(time.time() * 1000)
            }
            emission.update(pvs)
            iteration += 1
            i += 1
            yield emission
            time.sleep(self.emit_period)


if __name__ == "__main__":
    test_fast_pv = {'name': 'test_fast_pv', 'rate_hz': 120, 'drop_rate': 0.04}
    test_slow_pv = {'name': 'test_slow_pv', 'rate_hz': 0.2, 'drop_rate': 0.}
    test_pvs = [test_fast_pv, test_slow_pv]

    pvspoofer = PVSpoofer(test_fast_pv)
    t_start = time.time_ns()
    t_end = t_start + NUM_NANOSEC_IN_1_SEC
    emission = pvspoofer.emit_readings(t_start, t_end)
    emits = []
    for emit in emission:
        t = emit['timeStamp']['secondsPastEpoch'] * NUM_NANOSEC_IN_1_SEC
        t_ns = emit['timeStamp']['nanoseconds']
        emits.append(t + t_ns)

    emits = np.array(emits)

    k2egspoofer = K2EGSpoofer(pv_configs=test_pvs, n_emits=3, emit_rate_hz=1)

    emissions = []
    for emit in k2egspoofer():
        emissions.append(emit)

    for emission in emissions:
        print(emission['iteration'], len(emission['test_fast_pv']), len(emission['test_slow_pv']))
