from RL_Exercise4_TD.GridWorld import *
from RL_Exercise4_TD.Display_util import *


def init_policy_prob_distribution():
    """
    init policy prob distribution
    :return: None
    """
    # the prob distribution that action be taken under policy pi
    # use a data structure (5x7x8) array to represent the probability distribution under policy pi
    policy = np.zeros([H_grid, W_grid, len(A)])  # policy prob distribution, which will be in 5x7x8 size
    action_prob = [0, 0.25, 0, 0.5, 0, 0.25, 0, 0]  # [NW, N, NE, E, SE, S, SW, W]
    # 5x7x8
    for s in relevant_S:
        policy[s] = action_prob.copy()

    return policy.copy()


def policy_evaluation(policy):
    V = np.zeros([H_grid, W_grid])
    next_state_reward_probabilities = get_next_state_reward_probability()
    # relavant_S = [s for s in S if not s in obstacle_states and not s in terminal_states]
    theta = 0.001
    i = 0
    # A_allowed = [1, 3, 5, 7]
    # repeat when delta > theta
    while True:
        i += 1
        delta = 0
        for s in relevant_S:
            v = V[s]
            sum_over_action = 0
            for a in A:
                sum_over_next_state = 0
                for s_next, reward, probability in next_state_reward_probabilities[s][a]:
                    sum_over_next_state += probability * (reward + gamma * V[s_next])
                sum_over_action += policy[s][a] * sum_over_next_state
            V[s] = sum_over_action
            delta = max(delta, abs(V[s] - v))
        if delta < theta:
            break
    return V.copy()


policy_be_evaluated = init_policy_prob_distribution()
V = policy_evaluation(policy=policy_be_evaluated)
print("The state values of the initial policy:")
show_state_values(V)
