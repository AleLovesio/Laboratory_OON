import numpy as np


# function that returns the SNR
def to_snr(signal_power, noise_power):
    return 10 * np.log10(signal_power / noise_power)


# function that returns the distance between two points
def line_len(node_pos, second_node_pos):
    return np.sqrt((node_pos[0] - second_node_pos[0]) ** 2 + (node_pos[1] - second_node_pos[1]) ** 2)
