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
    global relevant_S
    relevant_S = [s for s in S if not (s in obstacle_states or s in terminal_states)]
    return grid_world


load_grid_world(grid_world_config)
