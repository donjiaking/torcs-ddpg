import numpy as np
import random
from collections import deque

"""
store and sample transitions
"""
class ReplayBuffer():
    def __init__(self, capacity) -> None:
        self.capacity = capacity
        self.size = 0

        self.buffer = deque() # implement buffer as a queue
    
    # store (s, a, r, s_1, done) into the buffer
    def store(self, s, a, r, s_1, done):
        transition = (s, a, r, s_1, done)

        if(self.size >= self.capacity):
            self.buffer.popleft()
            self.buffer.append(transition)
        else:
            self.size += 1
            self.buffer.append(transition)
    
    # sample a batch of (s, a, r, s_1, done)
    def sample(self, batch_size) -> list:
        random.seed()

        batch = []
        for _ in range(batch_size):
            batch.append(self.buffer[random.randint(0, min(batch_size, self.size)-1)])

        return batch
    
    def __len__(self):
        return len(self.buffer)

