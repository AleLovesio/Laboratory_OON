import pandas as pd
import numpy as np
import scipy.constants as const
import matplotlib.pyplot as plt
import json as json
import random

NUMBER_OF_CHANNELS = 10


class SignalInformation:

    def __init__(self, signal_power=None, path=None):
        if signal_power:
            self._signal_power = signal_power
        else:
            self._signal_power = 0
        self._noise_power = 0
        self._latency = 0
        if path:
            self._path = path
        else:
            self._path = []

    @property
    def signal_power(self):
        return self._signal_power

#    @signal_power.setter
#    def signal_power(self, signal_power):
#        self._signal_power = signal_power

    @property
    def noise_power(self):
        return self._noise_power

    @noise_power.setter
    def noise_power(self, noise_power):
        self._noise_power = noise_power

    @property
    def latency(self):
        return self._latency

    @latency.setter
    def latency(self, latency):
        self._latency = latency

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, path):
        self._path = path

#    def increase_signal_power(self, additional_signal_power):
#        self.signal_power = additional_signal_power + self.signal_power

    def increase_noise_power(self, additional_noise_power):
        self.noise_power = additional_noise_power + self.noise_power

    def increase_latency(self, additional_latency):
        self.latency = additional_latency + self.latency

    def update_path(self):
        self.path = self.path[1:]


class Lightpath(SignalInformation):
    def __init__(self, signal_power=None, path=None, channel=None):
        super().__init__(signal_power, path)
        if channel:
            self._channel = channel
        else:
            self._channel = 0
            # self._channel = 193.5e12

    @property
    def channel(self):
        return self._channel

#    @channel.setter
#    def channel(self, channel):
#        self._channel = channel


class Node:

    def __init__(self, node_data):
        self._label = node_data['label']
        self._position = node_data['position']
        self._connected_nodes = node_data['connected_nodes']
        self._successive = {}

    @property
    def label(self):
        return self._label

#    @label.setter
#    def label(self, label):
#        self._label = label

    @property
    def position(self):
        return self._position

#    @position.setter
#    def position(self, position):
#        self._position = position

    @property
    def connected_nodes(self):
        return self._connected_nodes

#    @connected_nodes.setter
#    def connected_nodes(self, connected_nodes):
#        self._connected_nodes = connected_nodes

    @property
    def successive(self):
        return self._successive

    @successive.setter
    def successive(self, successive):
        self._successive = successive

    def propagate(self, signal_information):
        if len(signal_information.path) > 1:
            line = self.successive[signal_information.path[:2]]
            signal_information.update_path()
            signal_information = line.propagate(signal_information)
        return signal_information


class Line:

    def __init__(self, node_data):
        self._label = node_data['label']
        self._length = node_data['length']
        self._successive = {}
        self._state = ["free"]*NUMBER_OF_CHANNELS  # or ["occupied"]

    @property
    def label(self):
        return self._label

#    @label.setter
#    def label(self, label):
#        self._label = label

    @property
    def length(self):
        return self._length

#    @length.setter
#    def length(self, length):
#        self._length = length

    @property
    def successive(self):
        return self._successive

    @successive.setter
    def successive(self, successive):
        self._successive = successive

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, state):
        self._state = state

    def latency_generation(self):
        return (3 * self.length) / (2 * const.speed_of_light)

    def noise_generation(self, signal_power):
        return 0.001 * signal_power * self.length

#    def propagate(self, signal_information):
#        signal_information.increase_noise_power(self.noise_generation(signal_information.signal_power))
#        signal_information.increase_latency(self.latency_generation())
#        self.successive[signal_information.path[0]].propagate(signal_information)
#        return signal_information  # to return once the propagation is finished

    def propagate(self, lightpath):
        lightpath.increase_noise_power(self.noise_generation(lightpath.signal_power))
        lightpath.increase_latency(self.latency_generation())
        self.successive[lightpath.path[0]].propagate(lightpath)
        if type(lightpath) == Lightpath:
            self.state[int(lightpath.channel[-1])] = "occupied"
        return lightpath  # to return once the propagation is finished


class Network:

    def __init__(self, json_data_file):
        self._nodes = {}
        self._lines = {}
        self._nodes_data = json.load(open(json_data_file, 'r'))
        self._weighted_paths = pd.DataFrame()
        self._route_space = pd.DataFrame()
        for key in self._nodes_data:
            node_pos = tuple(self._nodes_data[key]["position"])
            conn_nodes = self._nodes_data[key]["connected_nodes"]
            self._nodes[key] = Node({'label': key, 'position': node_pos, 'connected_nodes': conn_nodes})
            for second_node_str in conn_nodes:
                line_name = key+second_node_str
                second_node_pos = self._nodes_data[second_node_str]["position"]
                line_length = np.sqrt((node_pos[0] - second_node_pos[0])**2 + (node_pos[1] - second_node_pos[1])**2)
                self._lines[line_name] = Line({'label': line_name, 'length': line_length})
        self.connect()
        weighted_paths_path_col = []
        weighted_paths_latency_col = []
        weighted_paths_noise_col = []
        weighted_paths_snr_col = []
        df_indexes = []

        for weighted_paths_start_node in self.nodes.keys():
            for weighted_paths_end_node in self.nodes.keys():
                if weighted_paths_start_node != weighted_paths_end_node:
                    for weighted_paths_path in self.find_paths(weighted_paths_start_node, weighted_paths_end_node):
                        weighted_paths_path_str = ""
                        for weighted_paths_path_node in weighted_paths_path:
                            weighted_paths_path_str += weighted_paths_path_node + "->"
                        df_indexes.append(weighted_paths_path_str.replace("->", ""))
                        weighted_paths_path_col.append(weighted_paths_path_str[:-2])
                        weighted_paths_sig_inf = SignalInformation(1, weighted_paths_path)
                        weighted_paths_sig_inf = self.propagate(weighted_paths_sig_inf)
                        weighted_paths_latency_col.append(weighted_paths_sig_inf.latency)
                        weighted_paths_noise_col.append(weighted_paths_sig_inf.noise_power)
                        weighted_paths_snr_col.append(10 * np.log10(weighted_paths_sig_inf.signal_power
                                                                    / weighted_paths_sig_inf.noise_power))

        self._weighted_paths["Path"] = weighted_paths_path_col
        self._weighted_paths["Latency"] = weighted_paths_latency_col
        self._weighted_paths["Noise"] = weighted_paths_noise_col
        self._weighted_paths["SNR"] = weighted_paths_snr_col
        self._route_space["Path"] = weighted_paths_path_col
        self._weighted_paths.index = df_indexes
        for col_num in range(NUMBER_OF_CHANNELS):
            self._route_space["CH"+str(col_num)] = ["free"]*len(weighted_paths_path_col)
        self._route_space.index = df_indexes

    @property
    def nodes(self):
        return self._nodes

    @property
    def lines(self):
        return self._lines

    @property
    def weighted_paths(self):
        return self._weighted_paths

    @property
    def route_space(self):
        return self._route_space

    def connect(self):
        for node_name in self.nodes:
            for connected_node in self.nodes[node_name].connected_nodes:
                line_name = node_name + connected_node
                self.lines[line_name].successive[connected_node] = self.nodes[connected_node]
                self.nodes[node_name].successive[line_name] = self.lines[line_name]

    def find_paths(self, label_node1, label_node2):
        paths_dict = {0: label_node1}
        new_paths = 1
        level = 0
        while new_paths != 0:
            new_paths = 0
            paths_dict[level+1] = []
            for this_level_path in paths_dict[level]:
                if this_level_path[-1] != label_node2:
                    connected_lines = self.nodes[this_level_path[-1]].successive
                    for connected_line in connected_lines:
                        if connected_line[-1] not in this_level_path:
                            paths_dict[level+1].append(this_level_path+connected_line[-1])
                            if connected_line[-1] != label_node2:
                                new_paths += 1
            level += 1
        paths = []
        for i_level in range(level):
            for final_path in paths_dict[i_level+1]:
                if final_path[-1] == label_node2:
                    paths.append(final_path)
        return paths

    def propagate(self, signal_information):
        prop_signal_information = self.nodes[signal_information.path[0]].propagate(signal_information)
        return prop_signal_information

    def draw(self):
        for node_name in self.nodes:
            x1 = self.nodes[node_name].position[0]
            y1 = self.nodes[node_name].position[1]
            plt.plot(x1, y1, 'go', markersize=5)
            plt.text(x1, y1 + 12000, node_name)
            for connected_node_name in self.nodes[node_name].connected_nodes:
                x2 = self.nodes[connected_node_name].position[0]
                y2 = self.nodes[connected_node_name].position[1]
                plt.plot([x1, x2], [y1, y2], 'b')
        plt.title('Network graph')
        plt.show()

    def find_best_snr(self, label_start_node, label_end_node):
        paths = self.find_paths(label_start_node, label_end_node)
        best_snr_path = ""
        best_snr = float('-inf')
        for path in paths:
            test_snr = \
                self.weighted_paths.loc[self.weighted_paths["Path"] == path.replace("", "->")[2:-2]]["SNR"].tolist()[0]
        #    path_free = True
        #    for node_iter in range(len(path)-1):
        #        channel_free = False
        #        for channel_iter in self.lines[path[node_iter:node_iter + 2]].state:
        #            if channel_iter == "free":
        #                channel_free = True
        #        if not channel_free:
        #            path_free = False
        #    if (test_snr > best_snr) and path_free:
            if (test_snr > best_snr) and ("free" in self.route_space.loc[path].tolist()) :
                best_snr_path = path
                best_snr = test_snr
        return best_snr_path

    def find_best_latency(self, label_start_node, label_end_node):
        paths = self.find_paths(label_start_node, label_end_node)
        best_latency_path = ""
        best_latency = float('inf')
        for path in paths:
            test_latency = \
                self.weighted_paths.loc[self.weighted_paths["Path"] ==
                                        path.replace("", "->")[2:-2]]["Latency"].tolist()[0]
        #    path_free = True
        #    for node_iter in range(len(path) - 1):
        #        channel_free = False
        #        for channel_iter in self.lines[path[node_iter:node_iter + 2]].state:
        #            if channel_iter == "free":
        #                channel_free = True
        #        if not channel_free:
        #            path_free = False
        #    if (test_latency < best_latency) and path_free:
            if (test_latency < best_latency) and ("free" in self.route_space.loc[path].tolist()):
                best_latency_path = path
                best_latency = test_latency
        return best_latency_path

    def stream(self, stream_connections_list, pref=None):
        if pref:
            pref = "Latency"
        else:
            pref = "SNR"
        for stream_connection in stream_connections_list:
            if pref == "Latency":
                path = self.find_best_latency(stream_connection.input, stream_connection.output)
            else:
                path = self.find_best_snr(stream_connection.input, stream_connection.output)
            if path != "":
                # signal_information = SignalInformation(1, path)
                # signal_information = self.propagate(signal_information)

                first_available_channel = self.route_space.loc[path].tolist().index("free")-1  # find the first one
                lightpath = Lightpath(1, path, "CH"+str(first_available_channel))
                lightpath = self.propagate(lightpath)
                # self.route_space.loc[path]["CH"+str(first_available_channel)] = "occupied"
                for occupied_route_node in path:
                    paths_to_be_occupied = [path for path in self.route_space.index.tolist() if path.find(occupied_route_node) >= 0]
                    for path_to_be_occupied in paths_to_be_occupied:
                        self.route_space.loc[path_to_be_occupied, "CH" + str(first_available_channel)] = "occupied"
                    # self.route_space.loc[self.route_space.index.str.find(occupied_route_node), "CH"+str(first_available_channel)] = "occupied"

                # stream_connection.latency = signal_information.latency
                # stream_connection.snr = \
                #    (10 * np.log10(signal_information.signal_power / signal_information.noise_power))
                stream_connection.latency = lightpath.latency
                stream_connection.snr = \
                    (10 * np.log10(lightpath.signal_power / lightpath.noise_power))
            else:
                stream_connection.latency = 0
                stream_connection.snr = "None"
        return stream_connections_list


class Connection:

    def __init__(self, input, output, signal_power):
        self._input = input
        self._output = output
        self._signal_power = signal_power
        self._latency = 0.0
        self._snr = 0.0

    @property
    def input(self):
        return self._input

    @property
    def output(self):
        return self._output

    @property
    def signal_power(self):
        return self._signal_power

    @property
    def latency(self):
        return self._latency

    @property
    def snr(self):
        return self._snr

    @latency.setter
    def latency(self, latency):
        self._latency = latency

    @snr.setter
    def snr(self, snr):
        self._snr = snr


if __name__ == "__main__":
    network = Network("nodes.json")
#    print(network.weighted_paths)
#    print(network.find_best_snr("A", "E"))
#    print(network.find_best_latency("A", "E"))
#    network.draw()
    network_nodes_list = list(network.nodes.keys())
    connections_list = []
    for i in range(100):
        [start_node, end_node] = random.sample(network_nodes_list, 2)
        connections_list.append(Connection(start_node, end_node, 1))
    for analysis_type in ["Latency", "SNR"]:
        if analysis_type != "Latency":
            network = Network("nodes.json")
        network = Network("nodes.json")
        network.stream(connections_list, analysis_type)
        streams_list = []
        for connection in connections_list:
            if analysis_type == "Latency" and connection.latency != 0:
                streams_list.append(connection.latency)
            elif analysis_type == "SNR" and connection.snr != "None":
                streams_list.append(connection.snr)

        plt.figure()
        plt.hist(streams_list, bins=15)
        if analysis_type == "Latency":
            unit = "[s]"
        else:
            unit = "[dB]"
        plt.xlabel(analysis_type+" Range "+unit)
        plt.ylabel("Paths")
        plt.title("Path choice: "+analysis_type)
        plt.show()
