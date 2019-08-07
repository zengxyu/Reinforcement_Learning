from RL_Exercise3_MC.GridWorld import *
from RL_Exercise3_MC.Display_util import *


def init_policy_prob_distribution():
    """
    init policy prob distribution
    :return: None
    """
    # the prob distribution that action be taken under policy pi
    # use a data structure (5x7x8) array to represent the probability distribution under policy pi
    policy = np.zeros([H_grid, W_grid, len(A)])  # policy prob distribution, which will be in 5x7x8 size
    action_prob = [0.0, 0.25, 0.0, 0.5, 0.0, 0.25, 0.0, 0.0]  # [NW, N, NE, E, SE, S, SW, W]
    # 5x7x8
    for s in relevant_S:
        policy[s] = action_prob.copy()

    return policy.copy()


def policy_evaluation(first_visit):
    # policy to be evaluated
    policy = init_policy_prob_distribution()
    # an arbitrary state value function
    V = np.zeros([H_grid, W_grid])
    # Counts
    Counts = np.zeros_like(V)
    # an empty list, for all s in S, size (i,j)
    returns = [[[] for j in range(W_grid)] for i in range(H_grid)]
    # record
    # store the length of each episode
    ep_length_list = []
    # store the total reward of each episode
    ep_reward_list = []
    episodes_num = 10000
    for i in range(episodes_num):
        # record the whether it has occurred in this episode
        occurrence_record = np.zeros((H_grid, W_grid))
        s0 = relevant_S[np.random.randint(0, len(relevant_S))]
        # a is taken under policy
        episode = generate_one_episode(policy=policy, s_start=s0, a_start=None)
        ep_length_list.append(len(episode))
        ep_reward_list.append(np.sum(np.array(episode)[:, 0]))
        # for each state s appearing in the episode
        for j, [reward, state, action] in enumerate(episode):
            # if it is the last state in this episode, break
            if j == len(episode) - 1:
                break
            # if it is first-visit method and it is the first occurrence of s, then continue
            # or if it is every-visit method, then continue
            if (first_visit and occurrence_record[state] == 0) or (not first_visit):
                # episode[:, 0] represents reward column
                # reward_list contains the rewards after j step,
                # these rewards in the reward_list need to be added together to get the return
                reward_list = np.array(episode)[:, 0][j + 1:len(episode)]
                return_gain = 0
                # traverse inversely
                for reward in reward_list[::-1]:
                    # this loop are computed repetitively. It can be optimized
                    return_gain = reward + gamma * return_gain
                # Append R the Returns(s)
                returns[state[0]][state[1]].append(return_gain)
                # compute the V[s]
                V[state] = np.average(np.array(returns)[state])
                state_next = episode[j + 1][1]
                Counts[state_next] += 1
                occurrence_record[state] = 1
    return V.copy(), Counts.copy(), ep_length_list.copy(), ep_reward_list.copy()


if __name__ == "__main__":
    V, Counts, ep_length_list, ep_reward_list = policy_evaluation(first_visit=False)
    show_state_values(V)
    show_counts_of_state(Counts)
    print("The average length of episodes:", np.average(ep_length_list))
    print("The average total reward of episode:", np.average(ep_reward_list))
