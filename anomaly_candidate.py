from queue import PriorityQueue


class AnomalyCandidate:
    def __init__(self, slow_index: int, slow_time: int):
        """
        Anomaly candidate for use with CandidateBucket below and the
        buffer.Buffer.

        Parameters
        ----------
        slow_index : int
            The 'slow' time index for when the anomaly occurs.
        slow_time : int
            The corresponding slow time for when the anomaly occurs
            in nanoseconds since the epoch.
        """
        self._slow_index = slow_index
        self.slow_time = slow_time

        self.window = [-533, 533]

        self.fast_index = None
        self.score = None
        # more things here?

    def __str__(self):
        ss = 'AnomalyCandidate'
        ss += f"(slow_index={self.slow_index}"
        ss += f", slow_time={self.slow_time})"
        return ss

    def __repr__(self):
        return self.__str__()

    # the following are used in the priority queue
    def __eq__(self, other):
        return self.slow_index == other.slow_index

    def __lt__(self, other):
        return self.slow_index < other.slow_index

    def __gt__(self, other):
        return self.slow_index > other.slow_index
    # the above are used in the priority queue

    @property
    def slow_index(self):
        return self._slow_index

    @slow_index.setter
    def slow_index(self, new_index: int):
        self._slow_index = new_index

    @property
    def window_slice(self) -> list[int]:
        return [self._slow_index + x for x in self.window]

    def __iadd__(self, index_change: int):
        self.slow_index += index_change

    def __isub__(self, index_change: int):
        self.slow_index -= index_change


class CandidateBucket(PriorityQueue):
    @property
    def oldest_candidate_slow_index(self) -> int:
        if not self.empty():
            cand = self.get()
            index = cand.slow_index
            self.put(cand)
        else:
            index = -1
        return index

    def update_slow_indexes(self, index_change: int):
        candidates = []
        # pull the candidates off and change their indexes
        while not self.empty():
            cand = self.get()
            cand.slow_index += index_change
            candidates.append(cand)
        # put the candidates back on
        for candidate in candidates:
            self.put(candidate)


if __name__ == '__main__':
    bucket = CandidateBucket()

    # this will still work even though the bucket is empty
    bucket.update_slow_indexes(22)

    for i in range(5):
        bucket.put(AnomalyCandidate(i, i))

    print(bucket.queue)
    print(bucket.oldest_candidate_slow_index)
    item_0 = bucket.get()
    item_1 = bucket.get()
    print(bucket.oldest_candidate_slow_index)
    print(bucket.queue)
    bucket.put(item_0)
    print(bucket.queue)
    bucket.update_slow_indexes(-4)
    print(bucket.queue)

