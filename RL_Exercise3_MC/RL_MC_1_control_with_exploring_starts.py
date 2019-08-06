from RL_Exercise3_MC.GridWorld import *
from RL_Exercise3_MC.Display_util import *

"""
    # if we have the environmental model, then we know the transition probability and reward.
    # so when we compute state value, we can get action value, to optimize the policy.
    #
    # if we don't have the environmental model, we need to compute the action values.
    # if we start from start states,then some of the state-pair won't appear, so the solution is exploring start

"""


def mc_policy_iteration_with_exploring_starts(first_visit):
    # arbitrary action value
    Q = np.zeros((H_grid, W_grid, len(A)))
    # init policy
    policy = np.reshape([1 / len(A)] * H_grid * W_grid * len(A), (H_grid, W_grid, len(A)))
    # empty list
    returns = [[[[] for k in range(len(A))] for j in range(W_grid)] for i in range(H_grid)]

    # Following are for record
    # Initialize V, to record, used to decide whether converging
    V = np.zeros((H_grid, W_grid))
    # record counts
    Counts = np.zeros_like(V)
    # record counts for s,a pair
    QCounts = np.zeros_like(Q)
    # store the length of each episode
    ep_length_list = []
    # store the total reward of each episode
    ep_reward_list = []
    # record the max difference
    delta = 0
    # when the max difference is over theta, this means convergence
    theta = 0.01
    # iteration index
    i = 0
    print("Initial policy : ")
    show_compare_to_real_optimal_policy(policy, relevant_S)
    while True:
        i = i + 1
        occurrence_record = np.zeros((H_grid, W_grid))
        # choose s0 in S and a0 in A(s0), subject to all pairs have probability larger than 0,
        # generate an episode starting from s0,a0
        s0 = relevant_S[np.random.randint(0, len(relevant_S))]
        # a0 is random, but the other a in taken under policy
        a0 = A[np.random.randint(0, len(policy[s0]))]
        episode = generate_one_episode(policy=policy, s_start=s0, a_start=a0)

        ep_length_list.append(len(episode))
        ep_reward_list.append(np.sum(np.array(episode)[:, 0]))
        # for each pair s,a appearing in the episode
        for j, [r, s, a] in enumerate(episode):
            if j == len(episode) - 1:
                break
            # if it is first-visit method and it is the first occurrence of s, then continue
            # or if it is every-visit method, then continue
            if (first_visit and occurrence_record[s] == 0) or (not first_visit):
                v = V[s]
                reward_list = np.array(episode)[:, 0][j + 1:len(episode)]
                return_gain = 0
                # traverse inversely
                for reward in reward_list[::-1]:
                    # this loop are computed repetitively. It can be optimized
                    return_gain = reward + gamma * return_gain
                # Append R the Returns(s)
                returns[s[0]][s[1]][a].append(return_gain)
                # set Q[s,a] using the average of the returns[s,a]
                # Q[s][a] = np.average(np.array(returns)[s][a])
                # use incremental method to improve the running speed
                alpha = 1.0 / (QCounts[s][a] + 1.0)
                Q[s][a] = Q[s][a] + alpha * (return_gain - Q[s][a])
                s_next = episode[j + 1][1]
                # following are records
                Counts[s_next] += 1
                QCounts[s][a] += 1
                V[s] = np.max(Q[s])
                delta = max(delta, abs(V[s] - v))
                occurrence_record[s] = 1
        # update the policy, for each s in the episode
        for j, [r, s, a] in enumerate(episode):
            policy[s] = np.eye(len(A))[np.argmax(Q[s])]
        if i % 5000 == 0:
            print_iteration_information(i, delta < theta, delta, ep_length_list, ep_reward_list, V, Counts, policy,
                                        relevant_S)
            if delta < theta:
                break
            delta = 0

    return Q.copy(), policy.copy(), ep_length_list.copy(), ep_reward_list.copy()


Q, policy, ep_length_list, ep_reward_list = mc_policy_iteration_with_exploring_starts(first_visit=True)
print("The average length of episodes:", np.average(ep_length_list))
print("The average total reward of episode:", np.average(ep_reward_list))
show_optimal_policy(policy=policy, relevant_S=relevant_S)
