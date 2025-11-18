import numpy as np

grid = np.zeros((10, 15), dtype=int)

for i in range(10) :
    for j in range(15) :

        grid[i, j] = i+j

print(grid)