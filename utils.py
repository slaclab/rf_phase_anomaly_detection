from beam_check_config import NANOSECS_IN_1_SEC

def get_timestamp_ns(entry: dict) -> int:
    """
    Extract the timestamp in nanoseconds from a PV data-point from a snapshot entry.

    Parameters:
        entry (dict): A dictionary for a givin PV in a snapshot.
        (can get this by doing: `snapshot.get(pv, [])`)

    Returns:
        int: The timestamp of a PV data-point in nanoseconds
    """
    ts = entry.get("timeStamp", {})
    seconds = ts.get("secondsPastEpoch", 0)
    nanos = ts.get("nanoseconds", 0)
    return int(seconds * NANOSECS_IN_1_SEC + nanos)

def get_value(entry: dict) -> float:
    """
    Extract the value of a singe PV data-point from a snapshot entry.

    Parameters:
        entry (dict): A dictionary for a givin PV in a snapshot.
        (can get this by doing: `snapshot.get(pv, [])`)

    Returns:
        float: The PV data-point, or NaN if not valid in the snapshot entry.
    """
    if "value" in entry:
        return entry["value"]
    elif "index" in entry:
        return entry["index"]
    else:
        return float("nan")