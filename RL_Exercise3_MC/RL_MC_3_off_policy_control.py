from RL_Exercise3_MC.GridWorld import *
from RL_Exercise3_MC.Display_util import *


def off_policy_mc_control(first_visit):
    # behavior policy generates behavior in environment
    # estimation policy is policy being learned about
    # Weight returns from behavior policy by their relative probability of
    # occurring under the behavior and estimation policies

    # initialize
    epsilon = 0.1
    # action value
    Q = np.reshape([[-np.Inf] * (H_grid * W_grid * len(A))], [H_grid, W_grid, len(A)])
    # numerator
    N = np.reshape(np.zeros((H_grid * W_grid * len(A))), [H_grid, W_grid, len(A)])
    # Denominator of Q[s,a]
    D = np.reshape(np.zeros((H_grid * W_grid * len(A))), [H_grid, W_grid, len(A)])
    # an arbitrary deterministic policy

    target_policy = np.array(
        [[np.eye(len(A))[np.random.randint(0, len(A))] for j in range(W_grid)] for i in range(H_grid)])
    behavior_policy = np.reshape([1 / len(A)] * H_grid * W_grid * len(A), (H_grid, W_grid, len(A)))
    episodes_num = 100000
    for i in range(episodes_num):
        # print(i)
        # choose s0 in S and a0 in A(s0), subject to all pairs have probability larger than 0,
        # generate an episode starting from s0,a0
        occurrence_record = np.zeros((H_grid, W_grid))
        # choose s0 in S and a0 in A(s0), subject to all pairs have probability larger than 0,
        # generate an episode starting from s0,a0
        s0 = relevant_S[np.random.randint(0, len(relevant_S))]
        a0 = A[np.random.randint(0, len(behavior_policy[s0]))]
        episode = generate_one_episode(policy=behavior_policy, s_start=s0, a_start=a0)
        # tau <- last time at which a[tau] != target_policy[s[tau]]
        tau = len(episode) - 1
        for j, [r, s, a] in enumerate(episode[::-1]):
            if a is not None:
                if a == np.argmax(target_policy[s]):
                    tau = len(episode) - 1 - j
                else:
                    break
        tau = tau - 1
        episode_after_tau = episode[tau:len(episode)]
        # for each pair s,a appearing in the episode
        for j, [r, s, a] in enumerate(episode_after_tau):
            if j == len(episode_after_tau) - 1:
                break
                # if it is first-visit method and it is the first occurrence of s, then continue
                # or if it is every-visit method, then continue
            if (first_visit and occurrence_record[s] == 0) or (not first_visit):
                weight = 1
                s_a_pair_after_tau = np.array(episode_after_tau)[:, 1:3][j + 1:-1]
                for [s_, a_] in s_a_pair_after_tau:
                    weight = weight * behavior_policy[s_][a_]

                reward_list = np.array(episode)[:, 0][j + 1:len(episode)]
                return_gain = 0
                # traverse inversely
                for reward in reward_list[::-1]:
                    # this loop are computed repetitively. It can be optimized
                    return_gain = return_gain + gamma * reward
                N[s][a] = N[s][a] + weight * return_gain
                D[s][a] = D[s][a] + weight
                Q[s][a] = N[s][a] / D[s][a]
                occurrence_record[s] = 1
        # update the policy, for each s in the episode
        for j, [r, s, a] in enumerate(episode_after_tau):
            target_policy[s] = np.eye(len(A))[np.argmax(Q[s])]
    return Q, target_policy


Q, target_policy = off_policy_mc_control(first_visit=True)
show_optimal_policy(policy=target_policy, relevant_S=relevant_S)
