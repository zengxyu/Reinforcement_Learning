import numpy as np


def show_optimal_policy(policy, relevant_S):
    H_grid = np.shape(policy)[0]
    W_grid = np.shape(policy)[1]
    arrows = ['↖', '↑', '↗', '→', '↘', '↓ ', '↙', '←']
    optimal_policy_with_arrows = np.reshape([["●"] * (H_grid * W_grid)], [H_grid, W_grid])
    for s in relevant_S:
        action = np.argmax(policy[s])
        optimal_policy_with_arrows[s] = arrows[int(action)]

    print("Output optimal policy with arrows:")
    tplt = "{0:^8}"
    for i in range(np.shape(optimal_policy_with_arrows)[0]):
        for j in range(np.shape(optimal_policy_with_arrows)[1]):
            print(tplt.format(optimal_policy_with_arrows[(i, j)]), end=" ")
        print()
    print("Optimal policy", "==================================END")
    print()


def write_optimal_policy(policy, relevant_S):
    H_grid = np.shape(policy)[0]
    W_grid = np.shape(policy)[1]
    actions_taken = np.reshape([[-1] * (H_grid * W_grid)], [H_grid, W_grid])
    for s in relevant_S:
        actions_taken[s] = np.argmax(policy[s])
    np.savetxt("results/results_optimal_policy.txt", actions_taken, fmt="%d", delimiter=" ")


def print_episodes(episodes):
    for i in range(len(episodes)):
        print(i, "th episodes")
        print(episodes[i])
        print("++++++++++++++++++")


def show_state_values(V):
    H_grid = np.shape(V)[0]
    W_grid = np.shape(V)[1]
    print("State values: ")
    for i in range(H_grid):
        for j in range(W_grid):
            print("%8.3f" % V[(i, j)], end=" ")
        print()
    print("State values", "==================================END")
    print()


def show_counts_of_state(Counts):
    H_grid = np.shape(Counts)[0]
    W_grid = np.shape(Counts)[1]
    print("State Counts: ")
    for i in range(H_grid):
        for j in range(W_grid):
            print("%8.3f" % Counts[(i, j)], end=" ")
        print()
    print("State Counts", "==================================END")
    print()


def show_compare_to_real_optimal_policy(policy, relevant_S):
    H_grid = np.shape(policy)[0]
    W_grid = np.shape(policy)[1]
    optimal_actions_taken = np.loadtxt("results/results_optimal_policy.txt", delimiter=" ")
    actions_taken = np.reshape([[-1] * (H_grid * W_grid)], [H_grid, W_grid])
    for s in relevant_S:
        actions_taken[s] = np.argmax(policy[s])
    correct_elements = np.array(optimal_actions_taken) == np.array(actions_taken)
    # -11 is the number of obstacles
    correct_num = np.sum(correct_elements) - 11
    print("The number of correct policy:", correct_num, "/24")
    return correct_num


def print_iteration_information(i, converging, delta, ep_length_list, ep_reward_list, V, Counts, policy, relevant_S):
    print("Time step i: ", i, "; Converging or not: ", converging, ";delta: ", delta)
    print("The average length of episodes:", np.average(ep_length_list))
    print("The average total reward of episode:", np.average(ep_reward_list))
    show_compare_to_real_optimal_policy(policy=policy, relevant_S=relevant_S)
    show_state_values(V)
    show_counts_of_state(Counts)
    show_optimal_policy(policy=policy, relevant_S=relevant_S)
