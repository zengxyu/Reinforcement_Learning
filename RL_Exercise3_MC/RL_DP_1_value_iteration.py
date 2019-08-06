from RL_Exercise3_MC.GridWorld import *
from RL_Exercise3_MC.Display_util import *

"""
    knowing the full model of environment,
    so use dynamic programming to compute the real optimal policy and the real state value
    as to be reference to the following algorithm.
    
    We see immediately that this is much more plausible.
    The problem is, however, that value iteration is usually not an option in real-world applications.
"""






def value_iteration():
    """
    Compute the real optimal policy, to compare with following sarsa and Q-learning algorithm
    We see immediately that this is much more plausible.
    The problem is, however, that value iteration is usually not an option in real-world applications.
    :return:
    """
    # Initialize V, used to judge whether converged
    V = np.zeros([H_grid, W_grid])
    # Initialize Q(s,a) arbitrarily
    Q = np.reshape([0.0 for i in range(H_grid * W_grid * len(A))], [H_grid, W_grid, len(A)])
    # policy
    policy = np.reshape([1 / len(A)] * H_grid * W_grid * len(A), (H_grid, W_grid, len(A)))
    next_state_reward_probabilities = get_next_state_reward_probability()
    # value iteration algorithm
    theta = 0.001
    while True:
        delta = 0
        for s in relevant_S:
            v = V[s].copy()
            for a in A:
                sum_over_next_state = 0
                for s_next, reward, probability in next_state_reward_probabilities[s][a]:
                    sum_over_next_state += probability * (reward + gamma * V[s_next])
                Q[s][a] = sum_over_next_state
            # deterministic, because the V[s] are set to a deterministic maximal Q[s,a]
            V[s] = np.max(Q[s])
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
    # output a deterministic policy
    for s in relevant_S:
        action_argmax = np.argmax(Q[s])
        policy[s] = np.eye(len(A))[action_argmax]
    return policy.copy(), V.copy()



policy, V = value_iteration()
print("This is the real optimal policy, knowing the full model of environment")
show_optimal_policy(policy=policy, relevant_S=relevant_S)
print("The state values of optimal policy after value iteration:")
show_state_values(V)
