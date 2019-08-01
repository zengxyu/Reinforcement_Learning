from RL_Exercise4_MC.GridWorld import *
from RL_Exercise4_MC.Display_util import *


def mc_policy_iteration_with_exploring_starts(first_visit):
    # if we have the environmental model, then we know the transition probability and reward.
    # so when we compute state value, we can get action value, to optimize the policy.
    #
    # if we don't have the environmental model, we need to compute the action values.
    # if we start from start states,then some of the state-pair won't appear, so the solution is exploring start

    # arbitrary action value
    Q = np.reshape([[-np.Inf] * (H_grid * W_grid * len(A))], [H_grid, W_grid, len(A)])
    # init policy
    policy = np.reshape([1 / len(A)] * H_grid * W_grid * len(A), (H_grid, W_grid, len(A)))
    # empty list
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
            # if it is first-visit method and it is the first occurrence of s, then continue
            # or if it is every-visit method, then continue
            if (first_visit and occurrence_record[s] == 0) or (not first_visit):
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
            policy[s] = np.eye(len(A))[np.argmax(Q[s])]

    return Q.copy(), policy.copy()


Q, policy = mc_policy_iteration_with_exploring_starts(first_visit=True)
show_optimal_policy(policy=policy, relevant_S=relevant_S)
