import numpy as np

grid_world_config = "gridworld.txt"

W_grid = 9
H_grid = 9
grid_world = []
# actions
A = [0, 1, 2, 3, 4, 5, 6, 7]  # [NW, N, NE, E, SE, S, SW, W]
S = [(i, j) for i in range(W_grid) for j in range(H_grid)]  # 81 x 2

gamma = 0.9
moving_direction = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]

# different
terminal_states = []
obstacle_states = []
star_states = []
empty_states = []
relevant_S = []

policy = []  # policy prob distribution, which will be in 9x9x8 size

next_state_reward_probabilities = np.zeros([W_grid, H_grid, len(A)], dtype=object)

V = np.zeros([W_grid, H_grid])
Q = np.reshape([[-np.Inf] * (W_grid * H_grid * len(A))], [W_grid, H_grid, len(A)])


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
        elif np.array(grid_world)[s] == "x":
            obstacle_states.append(s)
        elif np.array(grid_world)[s] == "*":
            star_states.append(s)
        else:
            empty_states.append(s)
    return grid_world


def get_next_state_reward_probability():
    s_next = None
    reward_s_a_sn = None
    prob_s_a_sn = None
    # assign value to relevant_s, declare that it is global variable
    global relevant_S
    relevant_S = [s for s in S if not (s in obstacle_states or s in terminal_states)]

    for s in relevant_S:
        for a in A:
            next_state_reward_probability = []
            # when in state s and action taken is a , the next state is ? ,the reward is ?, the probability is ?
            attempt_s = (s[0] + moving_direction[a][0], s[1] + moving_direction[a][1])
            if is_attempt_leave_grid(attempt_s):
                s_next = s
                reward_s_a_sn = -5
                prob_s_a_sn = 1
            else:
                if attempt_s in terminal_states:
                    s_next = attempt_s
                    reward_s_a_sn = 100
                    prob_s_a_sn = 1
                elif attempt_s in star_states:
                    s_next = attempt_s
                    reward_s_a_sn = 5
                    prob_s_a_sn = 1
                elif attempt_s in obstacle_states:
                    s_next = s
                    reward_s_a_sn = -20
                    prob_s_a_sn = 1
                elif attempt_s in empty_states:
                    s_next = attempt_s
                    reward_s_a_sn = -1
                    prob_s_a_sn = 1
            next_state_reward_probability.append([s_next, reward_s_a_sn, prob_s_a_sn])
            # print(next_state_reward_probability)
            # next states can be multiple, when in the state s and taing action a, but in this game, next state is the only one
            # 9x9 x 8 x 1x3
            next_state_reward_probabilities[s][a] = next_state_reward_probability


def is_attempt_leave_grid(attempt_s):
    if attempt_s[0] < 0 or attempt_s[1] < 0 or attempt_s[0] > H_grid - 1 or attempt_s[1] > W_grid - 1:
        return True
    return False


load_grid_world(grid_world_config)
get_next_state_reward_probability()
