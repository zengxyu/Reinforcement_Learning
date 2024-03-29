from RL_Exercise2_DP.GridWorld import *

policy = np.zeros([W_grid, H_grid, len(A)])  # policy prob distribution, which will be in 9x9x8 size
four_A = [1, 3, 5, 7]


def get_next_state_reward_probability(deterministic):
    s_next = None
    reward_s_a_sn = None

    deviations = [0]
    probabilities = [1.0]

    # if the environmental modal is not deterministic, which means random next states will be resulted
    if not deterministic:
        deviations = [-1, 0, 1]
        probabilities = [0.15, 0.7, 0.15]

    # assign value to relevant_s, declare that it is global variable

    for s in relevant_S:
        for a in A:
            next_state_reward_probability = []
            # when in state s and action taken is a , the next state is ? ,the reward is ?, the probability is ?
            for i in range(len(deviations)):
                a_d = (a + deviations[i]) % 8
                attempt_s = (s[0] + moving_direction[a_d][0], s[1] + moving_direction[a_d][1])
                if is_attempt_leave_grid(attempt_s):
                    s_next = s
                    reward_s_a_sn = -5
                else:
                    if attempt_s in terminal_states:
                        s_next = attempt_s
                        reward_s_a_sn = 100
                    elif attempt_s in star_states:
                        s_next = attempt_s
                        reward_s_a_sn = 5
                    elif attempt_s in obstacle_states:
                        s_next = s
                        reward_s_a_sn = -20
                    elif attempt_s in empty_states:
                        s_next = attempt_s
                        reward_s_a_sn = -1
                next_state_reward_probability.append([s_next, reward_s_a_sn, probabilities[i]])
            # print(next_state_reward_probability)
            # next states can be multiple, when in the state s and taing action a,
            # but in this game, next state is the only one
            # 9x9 x 8 x 1x3
            next_state_reward_probabilities[s][a] = next_state_reward_probability


def is_attempt_leave_grid(attempt_s):
    if attempt_s[0] < 0 or attempt_s[1] < 0 or attempt_s[0] > H_grid - 1 or attempt_s[1] > W_grid - 1:
        return True
    return False


def get_init_policy_prob_distribution():
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


def task01():
    print("Task 01 : ===========================Start.")
    get_next_state_reward_probability(True)
    get_init_policy_prob_distribution()
    policy_evaluation()
    show_state_values(V)


def task02():
    print("Task 02 : ===========================Start.")
    get_next_state_reward_probability(True)
    get_init_policy_prob_distribution()
    policy_iteration()
    show_optimal_policy()


def task03():
    print("Task 03 : ===========================Start.")
    get_next_state_reward_probability(True)
    get_init_policy_prob_distribution()
    value_iteration()
    show_state_values(V)
    show_optimal_policy()


def task04():
    print("Task 04 : ===========================Start.")
    # non deterministic environmental modal
    # Specifically, the agent moves with probability 0.7 into the desired direction, but with probability 0.15
    # deviates 45° to the left and with probability 0.15 deviates 45° to the right of the desired direction.
    get_next_state_reward_probability(False)
    get_init_policy_prob_distribution()
    value_iteration()
    show_state_values(V)
    show_optimal_policy()


if __name__ == "__main__":
    # run the following four methods respectively
    # task01()
    # task02()
    # task03()
    task04()
