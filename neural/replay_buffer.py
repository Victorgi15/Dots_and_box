import random
from collections import deque
from typing import Deque, Iterable, List, Tuple


Sample = Tuple[List[List[List[float]]], List[float], float]


class ReplayBuffer:
    def __init__(self, capacity: int = 100000) -> None:
        self.capacity = capacity
        self._data: Deque[Sample] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._data)

    def add(self, sample: Sample) -> None:
        self._data.append(sample)

    def extend(self, samples: Iterable[Sample]) -> None:
        for sample in samples:
            self._data.append(sample)

    def sample(self, batch_size: int) -> List[Sample]:
        batch_size = min(batch_size, len(self._data))
        return random.sample(list(self._data), batch_size)
