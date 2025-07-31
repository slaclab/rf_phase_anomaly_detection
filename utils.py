from beam_check_config import NANOSECS_IN_1_SEC

def get_timestamp_ns(entry: dict) -> int:
    ts = entry.get("timeStamp", {})
    seconds = ts.get("secondsPastEpoch", 0)
    nanos = ts.get("nanoseconds", 0)
    return int(seconds * NANOSECS_IN_1_SEC + nanos)

def get_value(entry: dict) -> float:
    if "value" in entry:
        return entry["value"]
    elif "index" in entry:
        return entry["index"]
    else:
        return float("nan")