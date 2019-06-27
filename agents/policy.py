import copy
import numpy as np
import ipdb
from abc import ABCMeta, abstractmethod

class Policy(metaclass=ABCMeta):

    @abstractmethod
    def select_action(self, **kwargs):
        pass

class EpsGreedyQPolicy(Policy):
    """
        ε-greedy選択 
    """
    def __init__(self, epsilon=.1, decay_rate=1):
        super(EpsGreedyQPolicy, self).__init__()
        self.epsilon = epsilon
        self.decay_rate = decay_rate
        self.name = "epsilon-greedy"

    def select_action(self, q_values):
        assert q_values.ndim == 1
        nb_actions = q_values.shape[0]

        if np.random.uniform() < self.epsilon:  # random行動
            action = np.random.random_integers(0, nb_actions-1)
        else:   # greedy 行動
            action = np.argmax(q_values)

        return action

    def select_greedy_action(self, q_values):
        assert q_values.ndim == 1
        nb_actions = q_values.shape[0]
        action = np.argmax(q_values)

        return action
