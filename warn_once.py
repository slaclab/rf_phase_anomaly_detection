"""
Tools for warning only once in a while about k2eg issues.
"""

from dataclasses import dataclass


@dataclass
class WarnOnceInstance:
    pv_name: str
    reason: str
    iteration: int


class WarnOnceSet:
    def __init__(self, keep_for_this_many_iterations: int = 120):
        self.keep_for_this_many_iterations = keep_for_this_many_iterations
        self.warn_set = {}

    def __len__(self):
        return len(self.warn_set)

    def __contains__(self, pv_name: str):
        return pv_name in self.warn_set

    def add(self, pv_name: str, reason: str, iteration: int):
        self.warn_set[pv_name] = WarnOnceInstance(pv_name, reason, iteration)

    def remove(self, pv_name: str) -> WarnOnceInstance:
        return self.warn_set.pop(pv_name, None)

    def prune_old_instances(self, current_iteration: int) -> list[WarnOnceInstance]:
        removed_instances = []
        for instance in self.warn_set.values():
            if instance.iteration + self.keep_for_this_many_iterations <= current_iteration:
                removed_instances.append(instance)
        for instance in removed_instances:
            del self.warn_set[instance.pv_name]
        return removed_instances
