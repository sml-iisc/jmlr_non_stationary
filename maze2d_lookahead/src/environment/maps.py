import numpy as np

def get_map(map_rows, map_cols, N_gravity_centers, index=0):
    seed = 2025 + index
    np.random.seed(seed)
    MAP_ROWS = map_rows
    MAP_COLS = map_cols
    custom_map = [[0 for i in range(MAP_COLS)] for j in range(MAP_ROWS)]

    for i in range(MAP_ROWS):
        for j in range(MAP_COLS):
            if i==0 or j==0 or i==MAP_ROWS-1 or j==MAP_COLS-1:
                custom_map[i][j] = 1

    gravity_centers = np.random.uniform([1.5-MAP_COLS/2, 1.5-MAP_ROWS/2], [MAP_COLS/2-1.5, MAP_ROWS/2-1.5], size=(N_gravity_centers,2))
    # gravity_centers = np.array([[0.0, 0.0]])

    return custom_map, gravity_centers
