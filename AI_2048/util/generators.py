from collections import deque
from random import shuffle
class RL_sequence():
    def __init__(self, memory:deque, batch_size:int):
        self.memory = memory
        self.batch_size = batch_size
    def __len__(self):
        return math.ceil(len(self.memory) / self.batch_size)
    def __getitem__(self,idx):
        mini_batch = np.array(random.sample(samples, self.batch_size))
        update_in = np.array([x[0] for x in mini_batch])
        target = np.array([x[1] for x in mini_batch])
        return update_in, target
    def extract_validation_set(self,size):
        # Not runtime efficient
        shuffle(self.memory)
        v_set = [self.memory.pop() for _ in range(size)]
        return v_set