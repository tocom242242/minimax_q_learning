import numpy as np
import pulp
import sys

class MiniMaxQLearner():
    def __init__(self, 
                 aid=None, 
                 alpha=0.1, 
                 policy=None, 
                 gamma=0.99, 
                 ini_state="nonstate", 
                 actions=None):

        self.aid = aid
        self.alpha = alpha
        self.gamma = gamma
        self.policy = policy
        self.actions = actions
        self.state = ini_state
        self.q = {}
        self.q[ini_state] = {}
        self.pi = {}
        self.pi[ini_state] = np.repeat(1.0/len(self.actions), len(self.actions))
        self.v = {}

        self.previous_action = None
        self.reward_history = []
        self.pi_history = []

    def act(self, training=True):
        if training:
            action_id = self.policy.select_action(self.pi[self.state])
            action = self.actions[action_id]
            self.previous_action = action
        else:
            action_id = self.policy.select_greedy_action(self.pi)
            action = self.actions[action_id]

        return action

    def observe(self, state="nonstate", reward=None, opponent_action=None, is_learn=True):
        if is_learn:
            self.check_new_state(state)
            self.learn(state, reward, opponent_action)

    def learn(self, state, reward, opponent_action):
        self.reward_history.append(reward)
        self.q[state][(self.previous_action, opponent_action)] = self.compute_q(state, reward, opponent_action)
        self.pi[state][0],self.pi[state][1] = self.compute_pi()
        self.v[state] = self.compute_v(state)

        self.pi_history.append(self.pi[state][0])

    def compute_q(self, state, reward, opponent_action):
        if (self.previous_action, opponent_action) not in self.q[state].keys():
            self.q[state][(self.previous_action, opponent_action)] = 0.0
        q = self.q[state][(self.previous_action, opponent_action)]
        if state not in self.v.keys():
            self.v[state] = 0
        updated_q = q + (self.alpha * (reward+ self.gamma*self.v[state]- q))

        return updated_q

    def compute_v(self, state):
        min_expected_value = sys.maxsize
        for action2 in self.actions:
            expected_value = sum([self.pi[state][action1]*self.q[state][(action1, action2)] for action1 in self.actions])
            if expected_value < min_expected_value:
                min_expected_value = expected_value

        return min_expected_value

    def compute_pi(self):
        pi = pulp.LpVariable.dicts("pi",range(2), 0, 1)
        max_min_value = pulp.LpVariable("max_min_value")
        lp_prob = pulp.LpProblem("Maxmin Problem", pulp.LpMaximize)
        lp_prob += (max_min_value, "Objective")

        lp_prob += (pi[0]+pi[1] == 1)
        for action2 in self.actions:
            label = "{}".format(action2)
            values = pulp.lpSum([pi[idx] * self.q[self.state][(action1, action2)] for idx, action1 in enumerate(self.actions)])
            conditon = max_min_value <= values
            lp_prob += conditon

        lp_prob.solve()

        return pi[0].value(), pi[1].value()

    def check_new_state(self, state):
        if state not in self.q.keys():
            self.q[state] = {}

        for action1 in self.actions:
            for action2 in self.actions:
                if state not in self.pi.keys():
                    self.pi[state] = np.repeat(1.0/len(self.actions), len(self.actions))
                    self.v[state] = np.random.random()
                if (action1, action2) not in self.q[state].keys():
                    self.q[state][(action1, action2)] = np.random.random()
