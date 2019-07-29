import numpy as np


class Bandit:
    def __init__(self, id, interval, times=0, q=0, s=0):
        """
        construct a bandit with following properties
        :param id: id
        :param t: times that this bandit is chosen
        :param q: the action value which can predict the reward
        :param s: sum of rewards that this bandit recieved
        :param low_r: (low_r,up_r), the bound this bandit sample its reward uniformly from
        :param up_r:
        """
        self.id = id
        self.interval = interval
        self.times = times
        self.q = q
        self.s = s

    def set_interval(self, interval):
        self.interval = interval

    def get_interval(self):
        return self.interval

    def get_times(self):
        return self.times

    def get_reward(self):
        low_bound = self.interval[0]
        up_bound = self.interval[1]
        reward = np.random.rand() * (up_bound - low_bound) + low_bound
        return reward
