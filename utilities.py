

def read_pv_list_from_file(pv_list_file: str) -> list[str]:
    with open(pv_list_file, "r") as f:
        return [u for u in f.read().splitlines() if not u.startswith("#")]
