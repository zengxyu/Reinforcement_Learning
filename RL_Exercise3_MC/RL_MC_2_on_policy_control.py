from RL_Exercise3_MC.GridWorld import *
from RL_Exercise3_MC.Display_util import *


def on_policy__mc_control(first_visit):
    # in the realistic environment, it is impossible to start from a exploring start state and action
    # how to get rid of exploring starts.
    # e-greedy policy works

    # initialize
    epsilon = 0.1
    Q = np.reshape([[-np.Inf] * (H_grid * W_grid * len(A))], [H_grid, W_grid, len(A)])
    policy = np.reshape([1 / len(A)] * H_grid * W_grid * len(A), (H_grid, W_grid, len(A)))
    returns = [[[[] for k in range(len(A))] for j in range(W_grid)] for i in range(H_grid)]

    for i in range(episodes_num):
        occurrence_record = np.zeros((H_grid, W_grid))
        while True:
            # choose s0 in S and a0 in A(s0), subject to all pairs have probability larger than 0,
            # generate an episode starting from s0,a0
            s0 = relevant_S[np.random.randint(0, len(relevant_S))]
            a0 = A[np.random.randint(0, len(policy[s0]))]

            episode = generate_one_episode(policy=policy, s_start=s0, a_start=a0)
            # if the generated episode is none, regenerate the episode
            if episode is not None:
                break
        # for each pair s,a appearing in the episode
        for j, [r, s, a] in enumerate(episode):
            if j == len(episode) - 1:
                break
            if (first_visit and occurrence_record[s] == 0) or (not first_visit):
                # if it is first-visit method and it is the first occurrence of s, then continue
                # or if it is every-visit method, then continue
                reward_list = np.array(episode)[:, 0][j + 1:len(episode)]
                return_gain = 0
                # traverse inversely
                for reward in reward_list[::-1]:
                    # this loop are computed repetitively. It can be optimized
                    return_gain = return_gain + gamma * reward
                # Append R the Returns(s)
                returns[s[0]][s[1]][a].append(return_gain)
                # set Q[s,a] using the average of the returns[s,a]
                Q[s][a] = np.average(np.array(returns)[s][a])
                occurrence_record[s] = 1
        # update the policy, for each s in the episode
        for j, [r, s, a] in enumerate(episode):
            a_best = np.argmax(Q[s])
            for a in A:
                if a == a_best:
                    policy[s][a] = 1 - epsilon + epsilon / len(A)
                else:
                    policy[s][a] = epsilon / len(A)
    return Q, policy


Q, policy = on_policy__mc_control(first_visit=True)
show_optimal_policy(policy=policy, relevant_S=relevant_S)
