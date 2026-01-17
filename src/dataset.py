import numpy as np
from copy import deepcopy


class Dataset:
    def __init__(self, data, targets, count):
        self.data = data
        self.targets = targets
        self.count = count
        self.pos = 0
        self.batch_size = 0

    def nextBatch(self, size):
        if self.pos == self.count:
            self.randomShuffle()
            self.pos = 0

        if self.pos + size > self.count:
            self.pos = self.count - size

        input_data, output_data = self.data[self.pos: self.pos + size], self.targets[self.pos: self.pos + size]
        self.pos += size
        return input_data, output_data

    def randomShuffle(self):
        perms = np.random.permutation(self.count)
        self.apply_permuation(self.data, perms)
        self.apply_permuation(self.targets, perms)

    def apply_permuation(self, A, perms):
        perm = deepcopy(perms)
        for i in range(len(A)):
            while perm[i] != i:
                tmp = np.copy(A[perm[i]])
                A[perm[i]] = A[i]
                A[i] = tmp
                # A[perm[i]], A[i] = A[i], A[perm[i]] # TODO: why is this not working?
                perm[perm[i]], perm[i] = perm[i], perm[perm[i]]
