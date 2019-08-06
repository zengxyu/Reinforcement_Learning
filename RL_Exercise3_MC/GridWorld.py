import numpy as np

grid_world_config = "config/gridworld.txt"

H_grid = 5  # height
W_grid = 7  # width
grid_world = []
# actions
A = [0, 1, 2, 3, 4, 5, 6, 7]  # [NW, N, NE, E, SE, S, SW, W]
# states
S = [(i, j) for i in range(H_grid) for j in range(W_grid)]  # 35 x 2
# reward distribution
R_distributes = np.zeros((H_grid, W_grid))
# discounting factor
gamma = 0.9
# moving direction : current position plus moving direction is the next position.
# But in this grid world, transition probability should be taken into consideration
moving_direction = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1),
                    (0, -1)]  # [NW, N, NE, E, SE, S, SW, W]
deviations = [-1, 0, 1]  # transition direction, turn 45 left, No turn, turn 45 right
deviation_probabilities = [0.2, 0.6, 0.2]  # transition probability

# different types of state
terminal_states = []
obstacle_states = []
start_states = []
empty_states = []
# relevant_S include states not in terminal_states and obstacle_states
relevant_S = []


def load_grid_world(conf_file):
    # load grid world from configuration file
    with open(conf_file) as f:
        line_str = f.readline().strip()
        while line_str:
            line = list(filter(None, line_str.split(",")))  # delete the empty char
            grid_world.append(line)
            line_str = f.readline().strip()

    # analyze different states
    for s in S:
        if np.array(grid_world)[s] == "g":
            terminal_states.append(s)
        elif np.array(grid_world)[s] == "w":
            obstacle_states.append(s)
        elif np.array(grid_world)[s] == "s":
            start_states.append(s)
        else:
            empty_states.append(s)

    global relevant_S
    relevant_S = [s for s in S if not (s in obstacle_states or s in terminal_states)]
    return grid_world


def set_reward_distribution():
    for s in terminal_states:
        R_distributes[s] = 1000
    for s in obstacle_states:
        R_distributes[s] = -100
    for s in empty_states:
        R_distributes[s] = -1
    for s in start_states:
        R_distributes[s] = -1


def generate_one_episode(policy, s_start, a_start=None):
    episode = []
    cur_reward = 0
    cur_state = s_start
    i = 0
    while True:
        # if the current state is the terminal state, add the reward and the final state to the episode directly
        # then the episode ends
        if (cur_state in terminal_states) or (cur_state in obstacle_states) or len(episode) == 100:
            episode.append([cur_reward, cur_state, None])
            return episode
        # Why length can not be over 100?
        # if current state and current action is the same as the previous one state and previous one action,
        # it shows the policy definitely generate action that definitely leads to current state,
        # although deviation existing.
        # eg. state(0,0),if the policy generate action 0, even if plus the deviation (-1,0,1),
        # the moving direction 6,7,0(SW, W, NW) all will lead to state (0,0). So it is trapped into a infinite loop.
        # for episodes too long, the discounts will decay to be negligible,
        # and the Monte Carlo method must have an end(terminal state).
        # So restrict the length of episode to 100.

        cur_action = a_start if (i == 0 and a_start is not None) else get_action(policy, cur_state)
        # use a tuple like (cur_reward,cur_state,a) to form the episode
        episode.append([cur_reward, cur_state, cur_action])

        # now the current state and the current taken action is given, generate the next state
        s_next = get_next_state_deviated(cur_state, cur_action)
        r_next = R_distributes[s_next]

        cur_reward = r_next
        cur_state = s_next
        i += 1


def get_next_state_deviated(s, a):
    deviation = np.random.choice(a=deviations, p=deviation_probabilities)
    moving_direction_deviated = moving_direction[(a + deviations[deviation]) % len(A)]
    s_attempt_next = (s[0] + moving_direction_deviated[0], s[1] + moving_direction_deviated[1])
    s_next = get_valid_state(attempt_s=s_attempt_next)
    return s_next


def get_valid_state(attempt_s):
    """
    # truncate the coordinates to the valid coordinates
    # eg. (5,3)this coordinates is outside of the grid world, truncate (5,3) to (4,3)
    :param attempt_s:
    :return: s_next
    """
    s_next = (np.clip(attempt_s[0], 0, H_grid - 1), np.clip(attempt_s[1], 0, W_grid - 1))
    return s_next


def get_action(policy, s):
    action_indexes = np.arange(len(policy[s]))
    action = np.random.choice(a=action_indexes, p=policy[s])
    return action


def get_next_state_reward_probability():
    next_state_reward_probabilities = np.zeros([H_grid, W_grid, len(A)], dtype=object)
    for s in relevant_S:
        for a in A:
            next_state_reward_probability = []
            # when in state s and action taken is a , the next state is ? ,the reward is ?, the probability is ?
            for i in range(len(deviations)):
                a_d = (a + deviations[i]) % 8
                attempt_s = (s[0] + moving_direction[a_d][0], s[1] + moving_direction[a_d][1])
                s_next = get_valid_state(attempt_s=attempt_s)
                reward_s_a_sn = R_distributes[s_next]
                next_state_reward_probability.append([s_next, reward_s_a_sn, deviation_probabilities[i]])
            next_state_reward_probabilities[s][a] = next_state_reward_probability
    return next_state_reward_probabilities.copy()


load_grid_world(grid_world_config)
set_reward_distribution()
