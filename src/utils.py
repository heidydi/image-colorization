import numpy as np


def array2pics(data, dim_x, dim_y, channel=3):
    num_pics = int(len(data) / (dim_x * dim_y))
    images = np.zeros([num_pics, dim_x, dim_y, channel])
    idx = 0
    for x in range(num_pics):
        for i in range(dim_x):
            for j in range(dim_y):
                images[x][i][j] = data[idx]
                idx += 1
    return images
