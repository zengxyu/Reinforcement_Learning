import numpy as np

A = [0, 1, 2, 3, 4, 5, 6, 7]
H_grid = 5
W_grid = 7
print(A[0:-1])
# np.reshape([np.eye(len(A))[np.random.randint(0,len(A))]] * H_grid * W_grid, (H_grid, W_grid,len(A)))
# print(target_policy)
attempt_s = (-1,9)
s_next = (np.clip(attempt_s[0], 0, H_grid-1), np.clip(attempt_s[1], 0, W_grid-1))
print(s_next)