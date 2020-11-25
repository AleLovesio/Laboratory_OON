import random


# Function that adds "->" characters between the letters
def path_add_arrows(path):
    return path.replace("", "->")[2:-2]


def sample_2_nodes(network_nodes_list):
    return random.sample(network_nodes_list, 2)
