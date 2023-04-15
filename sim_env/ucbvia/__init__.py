import numpy as np


class SlotMachine(object):
    """ a simple slot machine """

    def __init__(self):
        """
        :param mean: actual win rate
        mean_estimate: current mean estimate
        N: idx of plays at the slot machine
        x: normalized reward
        """
        # self.mean = mean
        self.mean_estimate = 0
        self.N = 0
        self.x = 0

    def pull(self, x):
        """ modeled pull operation in a slot machine """
        self.x = x
        self.update()
        return self.x

    def update(self):
        """ updates mean estimate and tracks idx of pulls """
        self.N += 1
        self.mean_estimate = (1.0 - (1.0 / self.N)) * self.mean_estimate + (1.0 / self.N) * self.x

    def ucbvia(self, mean_estimate, total_num):
        if self.N == 0:
            return -float('inf')
        return mean_estimate - np.sqrt(2 * np.log(total_num) / self.N)

