from abc import ABC, abstractmethod
from multiprocessing import Process


class CustomProcessObject(ABC):
    @abstractmethod
    def __call__(self):
        raise NotImplementedError

    def begin(self):
        pass

    def end(self):
        pass


class ProcessManager:
    def __init__(self, process_objects: list[CustomProcessObject]):
        self.process_objects = process_objects
        self.processes = []
        self.is_running = False

    def __enter__(self):
        [po.begin() for po in self.process_objects]
        self.processes = [Process(target=po, args=()) for po in self.process_objects]
        [p.start() for p in self.processes]
        self.is_running = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        [p.join() for p in self.processes]
        [po.end() for po in self.process_objects]
        self.processes = []
        self.is_running = False
