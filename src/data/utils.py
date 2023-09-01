from scipy import signal
import numpy as np
import math


def rolling_median(data, kernel_size=15):
    """
    data: (T, V, C)
    size: window_size
    """
    temporal_last = data.transpose(1, 2, 0)  # V, C, T
    x_med = []
    output = []
    for x in temporal_last:
        x_med = [signal.medfilt(x_i, kernel_size=kernel_size) for x_i in x]
        x_med = np.stack(x_med)
        x_med = x_med.transpose(1, 0)
        output.append(x_med)
    output = np.stack(output).transpose(1, 0, 2)
    return output.astype(np.float32)


def rolling_mean(data, kernel_size=15):
    """
    data: (T, V, C)
    size: window_size
    """
    temporal_last = data.transpose(1, 2, 0)
    b = np.ones(kernel_size) / kernel_size
    output = []
    x_mean = []
    for x in temporal_last:
        x_mean = [signal.convolve(x_i, b, mode="same") for x_i in x]
        x_mean = np.stack(x_mean)
        x_mean = x_mean.transpose(1, 0)
        n_conv = math.ceil(kernel_size / 2)
        x_mean[0] *= kernel_size / n_conv
        for i in range(1, n_conv):
            x_mean[i] *= kernel_size / (i + n_conv)
            x_mean[-i] *= kernel_size / (i + n_conv - (kernel_size % 2))
        output.append(x_mean)
    output = np.stack(output).transpose(1, 0, 2)
    return output.astype(np.float32)
