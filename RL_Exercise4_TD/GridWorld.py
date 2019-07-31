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

policy = []  # policy prob distribution, which will be in 5x7x8 size

next_state_reward_probabilities = np.zeros([H_grid, W_grid, len(A)], dtype=object)

V = np.zeros([H_grid, W_grid])
Q = np.reshape([[-np.Inf] * (H_grid * W_grid * len(A))], [H_grid, W_grid, len(A)])

# the number of the episodes generated
episodes_num = 50000
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
        R_distributes[s] = -100
    for s in empty_states:
        R_distributes[s] = -1
    for s in start_states:
        R_distributes[s] = -1
    print(R_distributes)


load_grid_world(grid_world_config)
set_reward_distribution()
