import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from skimage import color
import scipy.misc


def load_origin_data(class_id=52):
    """load the cifar100 data and prepare input data set for neural net training"""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
    idxs = []
    for i in range(len(y_train)):
        if y_train[i] == [52]:
            idxs.append(i)
    # idxs = idxs[1:10]
    # the first 20 percent is chosen as validation data
    chosen_validation = x_train[idxs[0:int(len(idxs)/5)]]
    chosen_train = x_train[idxs[int(len(idxs)/5):]]

    idxs = []
    for i in range(len(y_test)):
        if y_test[i] == [52]:
            idxs.append(i)
    # idxs = idxs[1:10]
    chosen_test = x_test[idxs]
    return chosen_train, chosen_validation, chosen_test


def build_input(matrix, x, y, window_size):
    dim_x = len(matrix)
    dim_y = len(matrix[0])
    row = np.zeros((2 * window_size + 1) ** 2)
    idx = 0
    for i in range(x - window_size, x + window_size + 1):
        for j in range(y - window_size, y + window_size + 1):
            if i < 0 or i >= dim_x or j < 0 or j >= dim_y:
                row[idx] = 0.
            else:
                row[idx] = matrix[i][j]
            idx += 1
    return row


def prepare(data, window_size):
    """build the input data and targets for given data. each pixel has a corresponding data point"""
    num = len(data)
    dim = 32
    # plt.imshow(data[3])
    # plt.show()
    # scipy.misc.imsave("hehe.jpg", data[3])
    data = np.reshape(data, (-1, dim, dim, 3)) / 255.
    # scipy.misc.imsave("hehe_1.jpg", data[3])

    # plt.imshow(data[3])
    # plt.show()
    gray_data = color.rgb2gray(data)
    # scipy.misc.imsave("hehe_2.jpg", gray_data[3])
    # plt.imshow(gray_data[3])
    # plt.show()
    count = num * dim * dim  # number of data points
    input_data = np.zeros([count, (2 * window_size + 1) ** 2])
    targets = np.zeros([count, 3])
    # build input data and targets based on window size
    idx = 0
    for x in range(num):
        for i in range(dim):
            for j in range(dim):
                input_data[idx] = build_input(gray_data[x], i, j, window_size)
                targets[idx] = data[x][i][j]
                idx += 1

    return input_data, targets


def load(window_size=2, class_id=52):
    """prepare training data, validation data, and testing data"""
    chosen_train, chosen_validation, chosen_test = load_origin_data(class_id)
    # plt.imshow(chosen_test[0])
    # plt.show()
    # cao = color.rgb2gray(chosen_test[0])
    # plt.imshow(cao)
    # plt.show()
    train_input_data, train_targets = prepare(chosen_train, window_size)
    valid_input_data, valid_targets = prepare(chosen_validation, window_size)
    test_input_data, test_targets = prepare(chosen_test, window_size)

    return (train_input_data, train_targets), (valid_input_data, valid_targets), (test_input_data, test_targets)


if __name__ == '__main__':
    load(2)
