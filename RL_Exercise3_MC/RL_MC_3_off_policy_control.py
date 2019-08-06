from RL_Exercise3_MC.GridWorld import *
from RL_Exercise3_MC.Display_util import *

# set the format of print array
np.set_printoptions(formatter={'float': '{: 0.2f}'.format, 'int': '{:d}'.format})
"""
    # behavior policy generates behavior in environment
    # estimation policy is policy being learned about
    # Weight returns from behavior policy by their relative probability of
    # occurring under the behavior and estimation policies
"""


def off_policy_mc_control(first_visit):
    # initialize
    # action value
    Q = np.zeros((H_grid, W_grid, len(A)))
    # numerator
    N = np.zeros_like(Q)
    # Denominator of Q[s,a]
    D = np.zeros_like(Q)
    # an arbitrary deterministic policy, used to be the optimized
    target_policy = np.array(
        [[np.eye(len(A))[np.random.randint(0, len(A))] for j in range(W_grid)] for i in range(H_grid)])
    # used to generate the episodes
    behavior_policy = np.reshape([1 / len(A)] * H_grid * W_grid * len(A), (H_grid, W_grid, len(A)))

    # store the length of each episode
    ep_length_list = []
    # store the total reward of each episode
    ep_reward_list = []
    # Initialize V, to record, used to decide whether converging
    V = np.zeros((H_grid, W_grid))
    # record counts
    Counts = np.zeros_like(V)
    # record the max difference
    delta = 0
    # when the max difference is over theta, this means convergence
    theta = 0.1
    # iteration index
    i = 0
    # e-greedy params
    epsilon = 0.2
    print("Initial policy : ")
    show_compare_to_real_optimal_policy(target_policy, relevant_S)
    while True:
        i += 1
        occurrence_record = np.zeros((H_grid, W_grid))
        # choose s0 in S and a0 in A(s0), subject to all pairs have probability larger than 0,
        # generate an episode starting from s0,a0
        s0 = relevant_S[np.random.randint(0, len(relevant_S))]
        # a0 = A[np.random.randint(0, len(behavior_policy[s0]))]
        episode = generate_one_episode(policy=behavior_policy, s_start=s0, a_start=None)
        # tau <- last time at which a[tau] != target_policy[s[tau]]
        tau = len(episode) - 1
        for j, [r, s, a] in enumerate(episode[::-1]):
            a_max = np.argmax(target_policy[s])
            if a is not None:
                if a == a_max:
                    tau = len(episode) - 1 - j
                else:
                    break
        tau = tau - 1
        # record
        episode_after_tau = episode[tau:len(episode)]
        ep_length_list.append(len(episode_after_tau))
        ep_reward_list.append(np.sum(np.array(episode_after_tau)[:, 0]))
        for j, [r, s, a] in enumerate(episode_after_tau):
            if j == len(episode_after_tau) - 1:
                break
                # if it is first-visit method and it is the first occurrence of s, then continue
                # or if it is every-visit method, then continue
            if (first_visit and occurrence_record[s] == 0) or (not first_visit):
                v = V[s]
                weight = 1
                s_a_pair_after_j = np.array(episode_after_tau)[:, 1:3][j + 1:-1]
                for [s_, a_] in s_a_pair_after_j:
                    weight = weight * (1 / behavior_policy[s_][a_])

                reward_list = np.array(episode_after_tau)[:, 0][j + 1:len(episode_after_tau)]
                return_gain = 0
                # traverse inversely
                for reward in reward_list[::-1]:
                    # this loop are computed repetitively. It can be optimized
                    return_gain = reward + gamma * return_gain

                N[s][a] = N[s][a] + weight * return_gain
                D[s][a] = D[s][a] + weight
                Q[s][a] = N[s][a] / D[s][a]
                s_next = episode_after_tau[j + 1][1]
                # record
                Counts[s_next] += 1
                V[s] = np.max(Q[s])
                delta = max(delta, abs(V[s] - v))
                occurrence_record[s] = 1
        # update the policy, for each s in the episode
        for j, [r, s, a] in enumerate(episode_after_tau):
            target_policy[s] = np.eye(len(A))[np.argmax(Q[s])]
            a_best = np.argmax(Q[s])
            for a in A:
                if a == a_best:
                    behavior_policy[s][a] = 1 - epsilon + epsilon / len(A)
                else:
                    behavior_policy[s][a] = epsilon / len(A)
        if i % 50000 == 0:
            print_iteration_information(i, delta < theta, delta, ep_length_list, ep_reward_list, V, Counts,
                                        target_policy, relevant_S)
            # less than 1, regarded as converging. if theta is too small, a large number of loops are needed
            if delta < theta:
                break
            delta = 0
    return Q.copy(), target_policy.copy(), ep_length_list.copy(), ep_reward_list.copy()


Q, target_policy, ep_length_list, ep_reward_list = off_policy_mc_control(first_visit=False)
print("The average length of episodes:", np.average(ep_length_list))
print("The average total reward of episode:", np.average(ep_reward_list))
show_optimal_policy(policy=target_policy, relevant_S=relevant_S)
