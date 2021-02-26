import random
import os
import numpy as np
from datetime import datetime


# Function that adds "->" characters between the letters
def path_add_arrows(path):
    return path.replace("", "->")[2:-2]


# Get N random nodes from the node list
def sample_nodes(network_nodes_list, n_nodes=2):
    return random.sample(network_nodes_list, n_nodes)


# Method to update a route_space with updated nodes switching matrix and lines occupancy states
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


# Converts the line loss from linear to dB
def alpha_to_alpha_db(alpha):
    return alpha * 10 * np.log10(np.exp(1))


# Converts the line loss from dB to linear
def alpha_db_to_alpha(alpha_db):
    return alpha_db / (10 * np.log10(np.exp(1)))


# Create a uniform traffic matrix with requested bit rate equal to M*100Gbps
def generate_uniform_traffic_matrix(nodes_list, M):
    traffic_matrix = {}
    for node_x in nodes_list:
        traffic_matrix[node_x] = {}
        for node_y in nodes_list:
            if node_x == node_y:
                # No need to have traffic from node to itself
                traffic_matrix[node_x][node_y] = 0
            else:
                traffic_matrix[node_x][node_y] = M * 1e11
    # numpy array
    # return np.array([list(item.values()) for item in traffic_matrix.values()])
    # dictionary
    return traffic_matrix


# Get current time in hh:mm:ss string format
def current_time():
    return "[" + datetime.now().strftime("%H:%M:%S") + "] "


# Print on console with timestamp added at beginning of line
def print_with_time(string):
    print(current_time()+string)


# Add time to string at the beginning
def add_time(string):
    return current_time() + string


# Class used to log to file (and also print to console if needed)
class Logger:
    # constructor, opens the log file and sets the settings up
    def __init__(self, log_file_path=None, log_file_name=None, print_log_on_console=True, print_log_on_file=False):
        if (not log_file_name) or (not log_file_path):
            self.file = open(os.devnull, "w")
        else:
            self.file = open(log_file_path / log_file_name, 'w')
        self._print_log_on_console = print_log_on_console
        self._print_log_on_file = print_log_on_file

    # destructor, closes the file
    def __del__(self):
        self.file.close()

    @property
    def print_log_on_console(self):
        return self._print_log_on_console

    @property
    def print_log_on_file(self):
        return self._print_log_on_file

    # If called the line to be logged is also printed on the console thereafter
    def enable_log_on_console(self):
        self._print_log_on_console = True

    # If called the line to be logged will not be printed on the console thereafter
    def disable_log_on_console(self):
        self._print_log_on_console = False

    # If called the line to be logged will be written on file thereafter
    def enable_log_on_file(self):
        self._print_log_on_file = True

    # If called the line to be logged will not be written on file thereafter
    def disable_log_on_file(self):
        self._print_log_on_file = False

    # Logs the line according to the settings
    def log_line(self, line):
        if self._print_log_on_file:
            self.file.write(line+"\n")
        if self._print_log_on_console:
            print(line)

    # Logs the line with an additional timestamp at the beginning of the line
    def log_line_with_time(self, line):
        self.log_line(add_time(line))
