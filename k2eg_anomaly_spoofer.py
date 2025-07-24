import time

from k2eg_spoofer import PVSpoofer
from beam_check_config import SAMPLES_PER_SECOND, EXP_TMIT_MIN

from typing import Optional

n_periods = 15
n_points = n_periods * SAMPLES_PER_SECOND

constant_readings_dict = {
    'BPMS:IN20:221:TMITCUHBR': 3 * EXP_TMIT_MIN,
    'BPMS:LI24:801:XBR': 2.0,
    'BPMS:LTUH:250:XBR': 1.5,
    'BPMS:LTUH:450:XBR': -1.0,
    'BPMS:DMPH:502:YBR': 1.0,
    'BPMS:DMPH:693:YBR': -3.0,
    'BPMS:LTUH:250:TMITBR': 4 * EXP_TMIT_MIN,
    'BPMS:LTUH:450:TMITBR': 4 * EXP_TMIT_MIN,
    'BPMS:DMPH:502:TMITBR': 4 * EXP_TMIT_MIN,
    'BPMS:DMPH:693:TMITBR': 4 * EXP_TMIT_MIN,
    'IOC:BSY0:MP01:PC_RATE': 8,
    'IOC:IN20:EV01:RG02_ACTRATE': 10,
    'STPR:BSYH:2:STD2_IN_A': 0,
    'KLYS:LI20:61:PHAS_FASTBR': 25.,
    'KLYS:LI20:61:AMPL': 45.
}

def force_anomaly(constant_data: dict) -> dict:
    anom_data = {}
    for pv_name, reading_list in constant_data.items():
        if (
                pv_name.endswith('XBR') or
                pv_name.endswith('YBR') or
                pv_name.endswith('FASTBR') or
                pv_name.endswith('AMPL') or
                pv_name.endswith('TMITCUHBR') or
                pv_name.endswith('TMITBR')
        ):
            anom_start = len(reading_list) // 4
            anom_end = anom_start + len(reading_list) // 2
            anom_readings = []
            for i, reading in enumerate(reading_list):
                if (anom_start <= i and i < anom_end):
                    reading['value'] *= 20
                anom_readings.append(reading)
            anom_data[pv_name] = anom_readings
        else:
            anom_data[pv_name] = reading_list
    return constant_data


class K2EGAnomalySpoofer:
    def __init__(
            self,
            n_emits: Optional[int] = 0,
            emit_anomaly_every_n_iterations: Optional[int] = 6
    ) -> None:
        self.n_emits = n_emits
        self.emit_anomaly_every_n_iterations = emit_anomaly_every_n_iterations
        self.pv_configs = [
            {'name': name, 'rate_hz': 120, 'drop_rate': 0.}
            for name in constant_readings_dict.keys()
        ]
        self.emit_rate_hz = 1
        self.emit_period = 1 / self.emit_rate_hz

        self.pv_spoofers = {pvc['name']: PVSpoofer(pvc) for pvc in self.pv_configs}

    def __call__(self):
        iteration = 0
        if self.n_emits <= 0:
            i = int(-1e10)
        else:
            i = 0

        start_time = time.time_ns()
        while i < self.n_emits:
            # do some stuff
            pvs = {
                k: v.emit_readings(
                    start_time,
                    start_time + self.emit_period * 1e9,
                    constant_reading=constant_readings_dict[k]
                )
                for k, v in self.pv_spoofers.items()
            }
            if iteration % self.emit_anomaly_every_n_iterations == 0:
                pvs = force_anomaly(pvs)
            emission = {
                'iteration': iteration,
                'timestamp': int(time.time() * 1000)
            }
            emission.update(pvs)
            iteration += 1
            i += 1
            start_time += int(1e9)
            yield emission
            time.sleep(self.emit_period)