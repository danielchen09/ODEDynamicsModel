import numpy as np
from torch import nn


def one_hot(arr, size):
    if len(arr.shape) > 1:
        arr = arr[0]
    n = arr.shape[0]
    one_hot_encoding = np.zeros((n, size))
    one_hot_encoding[np.arange(n), arr] = 1
    return one_hot_encoding

def make_nn(features, final_activation=None):
    layers = []
    for i in range(len(features) - 1):
        layers.append(nn.Linear(features[i], features[i + 1]))
        if i < len(features) - 2:
            layers.append(nn.ReLU())
    if final_activation is not None:
        layers.apend(final_activation())
    return nn.Sequential(*layers)

if __name__ == '__main__':
    print(one_hot(np.array([2, 3, 0, 1, 2]), 5))