from RL_Exercise_TD.GridWorld import *

policy = np.zeros([H_grid, W_grid, len(A)])  # policy prob distribution, which will be in 9x9x8 size


def get_init_policy_prob_distribution():
    """
    :return: None
    """
    # the prob distribution that action be taken under policy pi
    # use a data structure (5x7x8) array to represent the probability distribution under policy pi
    action_prob = [0, 0.25, 0, 0.5, 0, 0.25, 0, 0]  # [NW, N, NE, E, SE, S, SW, W]
    # 5x7x8
    for s in relevant_S:
        policy[s] = action_prob.copy()


def generate_episodes():
    """
    # mc and td algorithm don't need a full environmental modal,not like dp,
    # the not-full environmental modal means it can be used to generate episodes,
    # but can't get the full prob distribution
    :return: episodes
    """

    episodes = []

    cur_reward = 0
    cur_state = start_states[0]

    for i in range(episodes_num):
        episode = []
        while True:
            # if the current state is the terminal state, add the reward and the final state to the episode directly
            # then the episode ends
            if (cur_state in terminal_states) or (cur_state in obstacle_states):
                episode.append((cur_reward, cur_state, None))
                episodes.append(episode.copy())
                # reset the variables
                episode.clear()
                cur_reward = 0
                cur_state = start_states[0]
                break
            # now the current state is cur_state, generate the action
            a = get_action(cur_state)
            # use a tuple like (cur_reward,cur_state,a) to form the episode
            episode.append((cur_reward, cur_state, a))

            # now the current start and the taken action is given, generate the next state
            moving_direction_deviated = get_moving_direction_deviated(a)
            s_attempt_next = (cur_state[0] + moving_direction_deviated[0], cur_state[1] + moving_direction_deviated[1])
            # if the s_attempt_next is in the outside of the grid world,
            # truncate the coordinates to the valid coordinates
            # (5,3)this coordinates is outside of the grid world, truncate (5,3) to (4,3)
            s_next = get_valid_state(attempt_s=s_attempt_next)
            r_next = R_distributes[s_next]

            cur_reward = r_next
            cur_state = s_next
    return episodes


def policy_evaluation():
    theta = 0.001
    i = 0
    # A_allowed = [1, 3, 5, 7]
    # repeat when delta > theta
    for episode in episodes:
        for i, (reward, state, action) in enumerate(episode):
            reward_next = episode[i + 1][0]
            state_next = episode[i + 1][1]
            V[state] = V[state] + alpha * (reward_next + gamma * V[state_next] - V[state])
            # if the next state is the last state , it is the terminal state
            if i + 1 == len(episode) - 1:
                if state_next in terminal_states:
                    print("有终止状态")
                break
    show_state_values(V)


def show_state_values(values):
    print("State values: ")
    for i in range(H_grid):
        for j in range(W_grid):
            print("%8.3f" % values[(i, j)], end=" ")
        print()
    print("State values", "==================================END")
    print()


def get_valid_state(attempt_s):
    """
    # truncate the coordinates to the valid coordinates
    # eg. (5,3)this coordinates is outside of the grid world, truncate (5,3) to (4,3)
    :param attempt_s:
    :return: s_next
    """
    s_next = attempt_s
    if attempt_s[0] < 0 or attempt_s[1] < 0:
        s_next = (max(attempt_s[0], 0), max(attempt_s[1], 0))
    if attempt_s[0] >= H_grid or attempt_s[1] >= W_grid:
        s_next = (min(attempt_s[0], H_grid - 1), min(attempt_s[1], W_grid - 1))
    return s_next


def get_moving_direction_deviated(a):
    """

    :param a:
    :return: moving_direction_deviated
    """
    deviation_index = np.random.randint(0, len(deviations))
    prob = deviation_probabilities[deviation_index]
    r = np.random.random()
    if r < prob:
        moving_direction_deviated = moving_direction[(a + deviations[deviation_index]) % 8]
    else:
        moving_direction_deviated = get_moving_direction_deviated(a)
    return moving_direction_deviated


def get_action(s):
    #
    action = np.random.randint(0, len(A))
    prob = policy[s][action]
    # if the probability of chosen action is 0, then get a new action again
    if prob == 0:
        return get_action(s)
    else:
        # if the probability of chosen action is not 0,
        # then consider whether to take this action according to its probability
        r = np.random.random()
        if r < prob:
            return action
        else:
            return get_action(s)


def print_episodes(episodes):
    for i in range(len(episodes)):
        print(i, "th episodes")
        print(episodes[i])
        print("++++++++++++++++++")


get_init_policy_prob_distribution()
episodes = generate_episodes()
policy_evaluation()
