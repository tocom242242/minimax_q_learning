import numpy as np
import matplotlib.pyplot as plt
from minimax_q_learner import MiniMaxQLearner 
from policy import EpsGreedyQPolicy
from matrix_game import MatrixGame

if __name__ == '__main__':
    nb_episode = 100

    agent1 = MiniMaxQLearner(aid=0, alpha=0.1, policy=EpsGreedyQPolicy(), actions=np.arange(2))  # agentの設定
    agent2 = MiniMaxQLearner(aid=1, alpha=0.1, policy=EpsGreedyQPolicy(), actions=np.arange(2))  # agentの設定

    game = MatrixGame()
    for episode in range(nb_episode):
        action1 = agent1.act()
        action2 = agent2.act()

        _, r1, r2 = game.step(action1, action2)

        agent1.observe(reward=r1, opponent_action=agent2.previous_action)
        agent2.observe(reward=r2, opponent_action=agent1.previous_action)
    print(agent1.pi)
    print(agent2.pi)
    # ipdb.set_trace()
    plt.plot(np.arange(len(agent1.pi_history)),agent1.pi_history, label="agent1's pi(0)")
    plt.plot(np.arange(len(agent2.pi_history)),agent2.pi_history, label="agent2's pi(0)")
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig("result.jpg")
    plt.show()
