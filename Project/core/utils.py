import random
import numpy as np
import scipy.constants as const

# Function that adds "->" characters between the letters
def path_add_arrows(path):
    return path.replace("", "->")[2:-2]


# Get two random nodes from the node list
def sample_2_nodes(network_nodes_list):
    return random.sample(network_nodes_list, 2)


# Method to update a route_space
def update_route_space(route_space, nodes, lines):
    for path in route_space.index.tolist():
        occupancy = np.array(route_space.loc[path].to_list()[1:])
        # update occupation with switching matrix
        for i in range(len(path)-2):
            occupancy = occupancy * np.array(nodes[path[i+1]].switching_matrix[path[i]][path[i+2]])
        # update occupation with line occupation
        for i in range(len(path)-1):
            occupancy = occupancy * np.array(lines[path[i:i+2]].state)
        for channel in range(len(occupancy)):
            route_space.loc[path, "CH" + str(channel)] = occupancy[channel]
    return route_space


def alpha_to_alpha_db(alpha):
    return alpha * 10 * np.log10(np.exp(1))


def alpha_db_to_alpha(alpha_db):
    return alpha_db / (10 * np.log10(np.exp(1)))
