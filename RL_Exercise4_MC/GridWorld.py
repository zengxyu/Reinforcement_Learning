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
# step size
alpha = 0.2
# there are 8 moving direction in each state s, this is different from actions in A,
# eg. If the action is 1, but the moving direction may be not (-1,0).
# It could be(-1, -1)or (-1, 0)or (-1, 1), which depends on the model of the environment
moving_direction = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1),
                    (0, -1)]  # [NW, N, NE, E, SE, S, SW, W]
deviations = [-1, 0, 1]
deviation_probabilities = [0.2, 0.6, 0.2]

# different types of state
terminal_states = []
obstacle_states = []
start_states = []
empty_states = []
relevant_S = []

# policy = []  # policy prob distribution, which will be in 5x7x8 size

# V = np.zeros([H_grid, W_grid])
# Q = np.reshape([[-np.Inf] * (H_grid * W_grid * len(A))], [H_grid, W_grid, len(A)])

# the number of the episodes generated
episodes_num = 10000


#


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
        R_distributes[s] = -10
    for s in empty_states:
        R_distributes[s] = -1
    for s in start_states:
        R_distributes[s] = -1
    print(R_distributes)


def generate_one_episode(policy, s_start, a_start=None):
    episode = []
    cur_reward = 0
    cur_state = s_start
    cur_action = a_start
    # if a_start == None
    if cur_action is None:
        cur_action = get_action(policy, cur_state)

    while True:
        # if the current state is the terminal state, add the reward and the final state to the episode directly
        # then the episode ends
        if (cur_state in terminal_states) or (cur_state in obstacle_states):
            episode.append([cur_reward, cur_state, None])
            return episode
        # if current state and current action is the same as the previous one state and previous one action,
        # it shows the policy definitely generate action that definitely leads to current state,
        # although deviation existing.
        # eg. state(1,0),if the policy generate action 7, although plus the deviation is (-1,0,1),
        # the moving direction 6,7,0 all will lead to state (1,0). So it is trapped into a infinite loop
        # to some extend, this is because insufficient exploration leading to wrong policy.
        # if (not len(episode) == 0) and (episode[len(episode) - 1][1] == cur_state) and (
        #         episode[len(episode) - 1][2] == cur_action):
        #     return None
        if len(episode) == 100:
            return None
        # now the current state is cur_state, generate the action

        # use a tuple like (cur_reward,cur_state,a) to form the episode
        episode.append([cur_reward, cur_state, cur_action])

        # now the current state and the current taken action is given, generate the next state
        moving_direction_deviated = get_moving_direction_deviated(cur_action)
        s_attempt_next = (cur_state[0] + moving_direction_deviated[0], cur_state[1] + moving_direction_deviated[1])
        # if the s_attempt_next is in the outside of the grid world,
        # truncate the coordinates to the valid coordinates
        # (5,3)this coordinates is outside of the grid world, truncate (5,3) to (4,3)
        s_next = get_valid_state(attempt_s=s_attempt_next)
        r_next = R_distributes[s_next]
        a_next = get_action(policy, cur_state)

        cur_reward = r_next
        cur_state = s_next
        cur_action = a_next


def get_valid_state(attempt_s):
    """
    # truncate the coordinates to the valid coordinates
    # eg. (5,3)this coordinates is outside of the grid world, truncate (5,3) to (4,3)
    :param attempt_s:
    :return: s_next
    """
    s_next = (np.clip(attempt_s[0], 0, H_grid-1), np.clip(attempt_s[1], 0, W_grid-1))
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


def get_action(policy, s):
    #
    # print(policy)
    action = np.random.randint(0, len(A))
    prob = policy[s][action]
    # if the probability of chosen action is 0, then get a new action again
    if prob == 0:
        return get_action(policy, s)
    else:
        # if the probability of chosen action is not 0,
        # then consider whether to take this action according to its probability
        r = np.random.random()
        if r < prob:
            return action
        else:
            return get_action(policy, s)


load_grid_world(grid_world_config)
set_reward_distribution()
