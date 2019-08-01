import numpy as np


def show_optimal_policy(policy, relevant_S):
    H_grid = np.shape(policy)[0]
    W_grid = np.shape(policy)[1]
    arrows = ['↖', '↑', '↗', '→', '↘', '↓ ', '↙', '←']
    optimal_policy_with_arrows = np.reshape([["●"] * (H_grid * W_grid)], [H_grid, W_grid])
    print("Optimal policy : ")
    print("policy shape : ", np.shape(policy))
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
