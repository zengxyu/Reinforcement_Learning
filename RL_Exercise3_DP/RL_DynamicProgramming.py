from GridWorld import *

policy = np.zeros([W_grid, H_grid, len(A)])  # policy prob distribution, which will be in 9x9x8 size
four_A = [1, 3, 5, 7]


def get_policy_prob_distribution():
    # the prob distribution that action be taken under policy pi
    # use a data structure (81x8) array to represent the probability distribution under policy pi
    # policy pi: action down : 0.5, up/left/right:0.5/3, others: 0
    action_prob = [0, 0.5 / 4, 0, 0.5 / 4, 0, 0.5 + 0.5 / 4, 0, 0.5 / 4]  # [NW, N, NE, E, SE, S, SW, W]
    # 9x9x8
    for s in relevant_S:
        policy[s] = action_prob.copy()


def policy_evaluation():
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
            delta = max(delta, np.abs(V[s] - v))
        if delta < theta:
            break

        # if i % 10 == 0:
        #     print()
        #     print("i == ", i, " 次迭代")
        #     print_values(V)


def policy_improvement():
    policy_stable = True
    # in task 2, the actions allowed to taken is up, right,down,left
    for s in relevant_S:
        b = policy[s].copy()
        for a in four_A:
            sum_over_next_state = 0
            for s_next, reward, probability in next_state_reward_probabilities[s][a]:
                sum_over_next_state += probability * (reward + gamma * V[s_next])
            Q[s][a] = sum_over_next_state
        # the index of maximum
        action_argmax = np.argmax(Q[s])
        # update the policy probability distribution when in the state s,
        # the sum of the probability is still 0, although only the action, that make Q(s,a) the largest, has the prob 1,
        # other actions are with prob 0,eg:[0,0,0,0,1,0,0]
        policy[s] = np.eye(len(A))[action_argmax]
        if not (b == policy[s]).all():
            policy_stable = False
    return policy_stable


def policy_iteration():
    # policy iteration algorithm, which includes two parts, one is policy evaluation, the other one is policy improvement
    # if the policy after improvement is different from the previous policy, it indicates that the policy is not stable
    # and it can be improved next
    k = 0
    while True:
        k += 1
        policy_evaluation()
        policy_stable = policy_improvement()
        if policy_stable:
            print("Policy stable")
            break
        else:
            print("Policy improvement the ", k, "th time !")
            show_state_values(V)
            continue


def value_iteration():
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


def value_iteration_with_non_deterministic_policy():
    # the value iteration algorithm with a non deterministic policy.
    theta = 0.01
    while True:
        delta = 0
        for s in relevant_S:
            v = V[s].copy()
            for a in A:
                sum_over_next_state = 0
                for s_next, reward, probability in next_state_reward_probabilities[s][a]:
                    sum_over_next_state += probability * (reward + gamma * V[s_next])
                Q[s][a] = sum_over_next_state
            # non deterministic
            argmax = np.argmax(Q[s])
            argmax_l45 = (argmax - 1) % len(A)
            argmax_r45 = (argmax + 1) % len(A)
            V[s] = 0.15 * Q[s][argmax_l45] + 0.7 * Q[s][argmax] + 0.15 * Q[s][argmax_r45]
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
    # output a non deterministic policy
    for s in relevant_S:
        argmax = np.argmax(Q[s])
        argmax_l45 = (argmax - 1) % len(A)
        argmax_r45 = (argmax + 1) % len(A)
        r = np.random.random()
        if r < 0.15:
            policy[s] = np.eye(len(A))[argmax_l45]
        elif 0.15 <= r < 0.85:
            policy[s] = np.eye(len(A))[argmax]
        else:
            policy[s] = np.eye(len(A))[argmax_r45]


def show_state_values(values):
    print("State values: ")
    for i in range(W_grid):
        for j in range(H_grid):
            print("%8.3f" % values[(i, j)], end=" ")
        print()
    print("State values", "==================================END")
    print()


def show_optimal_policy():
    arrows = ['↖', '↑', '↗', '→', '↘', '↓ ', '↙', '←']
    optimal_policy_with_arrows = np.reshape([["●"] * (9 * 9)], [W_grid, H_grid])
    print("Optimal policy : ")
    print("policy shape : ", np.shape(policy))
    for s in relevant_S:
        action = np.argmax(policy[s])
        optimal_policy_with_arrows[s] = arrows[action]

    print("Output optimal policy with arrows:")
    tplt = "{0:^8}"
    for i in range(np.shape(optimal_policy_with_arrows)[0]):
        for j in range(np.shape(optimal_policy_with_arrows)[1]):
            print(tplt.format(optimal_policy_with_arrows[(i, j)]), end=" ")
        print()
    print("Optimal policy", "==================================END")
    print()


get_policy_prob_distribution()


def task01():
    print("Task 01 : ===========================Start." )
    policy_evaluation()
    show_state_values(V)


def task02():
    print("Task 02 : ===========================Start.")
    policy_iteration()
    show_optimal_policy()


def task03():
    print("Task 03 : ===========================Start.")
    value_iteration()
    show_state_values(V)
    show_optimal_policy()


def task04():
    print("Task 04 : ===========================Start.")
    # non deterministic policy
    # Specifically, the agent moves with probability 0.7 into the desired direction, but with probability 0.15
    # deviates 45° to the left and with probability 0.15 deviates 45° to the right of the desired direction.
    value_iteration_with_non_deterministic_policy()
    show_state_values(V)
    show_optimal_policy()


if __name__ == "__main__":
    task01()
