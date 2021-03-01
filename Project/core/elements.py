from core import parameters as param
from core import science_utils as sci_util
from core import utils as util

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json as json
import copy


# Signal Information class: used to model a signal that traverses a path composed of lines and nodes.
class SignalInformation:
    # Constructor
    def __init__(self, signal_power=None, path=None):
        # Set signal power or set it zero if not defined
        if signal_power:
            self._signal_power = signal_power
        else:
            self._signal_power = 0
        # Initial noise set to zero, increased when the SignalInformation is propagated
        self._noise_power = 0
        # Initial latency set to zero, increased when the SignalInformation is propagated
        self._latency = 0
        # sets the path as an empty vector if not defined
        if path:
            self._path = path
        else:
            self._path = []
        # Default symbol rate
        self._Rs = param.Rs
        # Default frequency spacing between channels
        self._df = param.delta_f
        # Default ISNR equal to zero, gets updated after propagation
        self._ISNR = 0

    # Getters and setters
    @property
    def signal_power(self):
        return self._signal_power

    @signal_power.setter
    def signal_power(self, signal_power):
        self._signal_power = signal_power

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

    @property
    def Rs(self):
        return self._Rs

    @property
    def df(self):
        return self._df

    @property
    def ISNR(self):
        return self._ISNR

    @ISNR.setter
    def ISNR(self, ISNR):
        self._ISNR = ISNR

    # Method that is used to increase the noise power
    def increase_noise_power(self, additional_noise_power):
        self.noise_power = additional_noise_power + self.noise_power

    # Method that is used to increase the latency
    def increase_latency(self, additional_latency):
        self.latency = additional_latency + self.latency

    # Method used to remove the current node as we move forward in the propagation
    def update_path(self):
        self.path = self.path[1:]

    # Method used to set the launch power given in input to the current line
    def set_launch_power(self, launch_power):
        self.signal_power = launch_power

    # Method used to increase the ISNR as a line is traversed
    def increase_ISNR(self, additional_ISNR):
        self.ISNR = additional_ISNR + self.ISNR


# Class derived from the Signal Information class the is specific to a lightpath, a spectral channel property is added.
class Lightpath(SignalInformation):
    def __init__(self, signal_power=None, path=None, channel=None):
        # Call the constructor of the parent class, signal power and path set up there
        super().__init__(signal_power, path)
        # Sets channel or default channel
        if channel:
            self._channel = channel
        else:
            self._channel = "CH0"
            # self._channel = 193.5e12

    # Getter
    @property
    def channel(self):
        return self._channel


# Class that models an optical network node.
class Node:
    # Constructor, takes initialization data
    def __init__(self, node_data):
        # Node name
        self._label = node_data['label']
        # Node coordinate
        self._position = node_data['position']
        # Connected Nodes
        self._connected_nodes = node_data['connected_nodes']
        # Dictionary that links to connected lines
        self._successive = {}
        # Initial switching matrix
        self._switching_matrix = None
        # Transceiver type setting
        if 'transceiver' in node_data.keys():
            self._transceiver = node_data['transceiver']
        else:
            self._transceiver = "fixed_rate"

    # Setters and getters
    @property
    def label(self):
        return self._label

    @property
    def position(self):
        return self._position

    @property
    def connected_nodes(self):
        return self._connected_nodes

    @property
    def successive(self):
        return self._successive

    @successive.setter
    def successive(self, successive):
        self._successive = successive

    @property
    def switching_matrix(self):
        return self._switching_matrix

    @switching_matrix.setter
    def switching_matrix(self, switching_matrix):
        self._switching_matrix = switching_matrix

    @property
    def transceiver(self):
        return self._transceiver

    # Method that is used to forward the lightpath/signal information to the next line element,
    # setting its launch power, updating the path to be traveled, unless it's the last element.
    def propagate(self, lightpath):
        # check if it is not the last node
        if len(lightpath.path) > 1:
            # check if it is not a signal_information and if the next node switching matrix has to be updated
            # but only if the side channels have to be occupied
            if type(lightpath) == Lightpath and len(lightpath.path) > 2 and param.side_channel_occupancy:
                if (int(lightpath.channel[-1]) + 1) < param.NUMBER_OF_CHANNELS:
                    self.successive[lightpath.path[:2]].successive[lightpath.path[1]].switching_matrix[lightpath.path[
                        0]][lightpath.path[2]][int(lightpath.channel[-1]) + 1] = 0
                if (int(lightpath.channel[-1]) - 1) >= 0:
                    self.successive[lightpath.path[:2]].successive[lightpath.path[1]].switching_matrix[lightpath.path[
                        0]][lightpath.path[2]][int(lightpath.channel[-1]) - 1] = 0
            line = self.successive[lightpath.path[:2]]
            lightpath.set_launch_power(line.optimized_launch_power())
            lightpath.update_path()
            lightpath = line.propagate(lightpath)
        return lightpath


# Class that models an optical line.
class Line:
    # Constructor, takes initialization data
    def __init__(self, node_data):
        # Line name
        self._label = node_data['label']
        # Line length
        self._length = node_data['length']
        # Dictionary that links to the connected node
        self._successive = {}
        # vector that represent the spectral occupancy of the line
        self._state = [1] * param.NUMBER_OF_CHANNELS  # or [0]
        # Number of fiber spans in the line
        self._n_span = int(np.ceil(self._length / param.MAX_DISTANCE_BETWEEN_AMPLIFIERS))
        # Number of amplifiers in the line
        self._n_amplifiers = int(2 + np.floor(self._length / param.MAX_DISTANCE_BETWEEN_AMPLIFIERS))
        # Gain of the amplifiers
        if 'gain' in node_data.keys():
            self._gain = node_data['gain']
        else:
            self._gain = param.LINE_AMPLIFIER_GAIN
        # Noise figure of the amplifiers
        if 'nf' in node_data.keys():
            self._noise_figure = node_data['nf']
        else:
            self._noise_figure = param.LINE_AMPLIFIER_NF
        # Fiber loss
        if 'alpha' in node_data.keys():
            self._alpha = node_data['alpha']
        else:
            self._alpha = param.alpha_default
        # Dispersion
        if 'beta_2' in node_data.keys():
            self._beta_2 = node_data['beta_2']
        else:
            self._beta_2 = param.beta_2_default
        # nonlinearity coefficient
        if 'gamma' in node_data.keys():
            self._gamma = node_data['gamma']
        else:
            self._gamma = param.gamma_default
        self._L_eff = 1 / (2 * self._alpha)

    # Getters and setters
    @property
    def label(self):
        return self._label

    @property
    def length(self):
        return self._length

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

    @property
    def n_span(self):
        return self._n_span

    @property
    def n_amplifiers(self):
        return self._n_amplifiers

    @property
    def gain(self):
        return self._gain

    @gain.setter
    def gain(self, gain):
        self._gain = gain

    @property
    def alpha(self):
        return self._alpha

    @property
    def alpha_db(self):
        return util.alpha_to_alpha_db(self._alpha)

    @property
    def beta_2(self):
        return self._beta_2

    @property
    def gamma(self):
        return self._gamma

    @property
    def noise_figure(self):
        return self._noise_figure

    @property
    def L_eff(self):
        return self._L_eff

    # Methods that computes the added latency after travelling the line
    def latency_generation(self):
        return (3 * self.length) / (2 * param.c)

    # Method that computes the total noise due to non linear interference and amplified spontaneous emission
    def noise_generation(self, lightpath):
        # return 1e-9 * lightpath.signal_power * self.length
        # print("ASE: " + str(self.ase_generation()) + "  NLI: " + str(self.nli_generation(lightpath)))
        return self.ase_generation() + self.nli_generation(lightpath)

    # Method that forward the lightpath to the next node after increasing the lightpath noise power, latency and ISNR
    def propagate(self, lightpath):
        lightpath.increase_noise_power(self.noise_generation(lightpath))
        # lightpath.increase_ISNR(1 / sci_util.to_snr(lightpath.signal_power, self.noise_generation(lightpath)))
        lightpath.increase_ISNR(self.noise_generation(lightpath) / lightpath.signal_power)
        lightpath.increase_latency(self.latency_generation())
        self.successive[lightpath.path[0]].propagate(lightpath)
        if type(lightpath) == Lightpath:
            self.state[int(lightpath.channel[-1])] = 0
        return lightpath  # to return once the propagation is finished

    # Method used to compute the amplified spontaneous emission noise
    def ase_generation(self):
        return sci_util.ase(self.n_amplifiers, param.C_BAND_CENTER_FREQ, param.Bn, self.noise_figure, self.gain)

    # Method used to compute the non linear interference noise
    def nli_generation(self, lightpath):
        eta_nli = sci_util.nli_eta_nli(self.beta_2, lightpath.Rs, len(self.state),
                                       lightpath.df, self.gamma, self.alpha, self.L_eff)
        # print(eta_nli)
        return sci_util.nli(lightpath.signal_power, eta_nli, self.n_span, param.Bn)

    # Method that computes the optimal line launch power
    def optimized_launch_power(self, lightpath=None):
        if not lightpath:
            lightpath = Lightpath()
        return sci_util.opt_launch_pwr(self.ase_generation(), sci_util.nli_eta_nli(self.beta_2, lightpath.Rs,
                                                                                   len(self.state), lightpath.df,
                                                                                   self.gamma, self.alpha, self.L_eff),
                                       self.n_span, param.Bn)


# Class that models a Network composed of nodes (ROADMs) and lines (fiber lines)
class Network:
    # Constructor, takes initialization file in json format
    def __init__(self, json_data_file):
        # Network nodes dictionary
        self._nodes = {}
        # Network lines dictionary
        self._lines = {}
        # Opens json filed and converts it to dictionary
        self._nodes_data = json.load(open(json_data_file, 'r'))
        # Pandas dataframes for the weighted paths (latency, snr) and route space (channels occupancy in the network)
        self._weighted_paths = pd.DataFrame()
        self._route_space = pd.DataFrame()
        # Dictionary for the default switching matrix of the nodes
        self._default_switching_matrix_dict = {}
        # For each node in the file
        for key in self._nodes_data:
            # Take the position of the node
            node_pos = tuple(self._nodes_data[key]["position"])
            # Get the nodes connected to this node
            conn_nodes = self._nodes_data[key]["connected_nodes"]
            # Get transceiver type of this node, if not define set it as fixed rate
            if "transceiver" in self._nodes_data[key].keys():
                transceiver = self._nodes_data[key]["transceiver"]
            else:
                transceiver = "fixed_rate"
            # Create Node object for the currently parsed node
            self._nodes[key] = \
                Node({'label': key, 'position': node_pos, 'connected_nodes': conn_nodes, 'transceiver': transceiver})
            # If a switching matrix is given, it is set as the default one for the node
            if "switching_matrix" in self._nodes_data[key].keys():
                self._default_switching_matrix_dict[key] = self._nodes_data[key]["switching_matrix"]
            # For every connected node
            for second_node_str in conn_nodes:
                # Create the name of the line that connects the two nodes
                line_name = key + second_node_str
                # Get the second node position
                second_node_pos = self._nodes_data[second_node_str]["position"]
                # Compute the line length given the position of the two nodes
                line_length = sci_util.line_len(node_pos, second_node_pos)
                # Create the Line object
                self._lines[line_name] = Line({'label': line_name, 'length': line_length})
        # Now that all the Node and Line objects are created, link them
        self.connect()
        # Create the columns of the weighted paths dataframe
        weighted_paths_path_col = []
        weighted_paths_latency_col = []
        weighted_paths_noise_col = []
        weighted_paths_snr_col = []
        # Create vector of the indexes of the dataframe, will be the paths
        df_indexes = []
        # for a starting node within the nodes
        for weighted_paths_start_node in self.nodes.keys():
            # for another nodes within the nodes
            for weighted_paths_end_node in self.nodes.keys():
                # if it is not the same node
                if weighted_paths_start_node != weighted_paths_end_node:
                    # for a valid path between the nodes
                    for weighted_paths_path in self.find_paths(weighted_paths_start_node, weighted_paths_end_node):
                        # Add to the indexes the path (string)
                        df_indexes.append(weighted_paths_path)
                        # Add to the path column the string in the format START_NODE->NODE->END_NODE
                        weighted_paths_path_col.append(util.path_add_arrows(weighted_paths_path))
                        # Create a probing signal information
                        weighted_paths_sig_inf = SignalInformation(param.default_input_power, weighted_paths_path)
                        # Propagate the probing signal information
                        weighted_paths_sig_inf = self.propagate(weighted_paths_sig_inf)
                        # Add to the latency column this path latency
                        weighted_paths_latency_col.append(weighted_paths_sig_inf.latency)
                        # Add to the noise column this path noise
                        weighted_paths_noise_col.append(weighted_paths_sig_inf.noise_power)
                        # Add to the SNR column this path SNR in dB by inverting the retrieved ISNR
                        weighted_paths_snr_col.append(sci_util.linear_to_db(1 / weighted_paths_sig_inf.ISNR))
        # Add to the dataframe the columns of data
        self._weighted_paths["Path"] = weighted_paths_path_col
        self._weighted_paths["Latency"] = weighted_paths_latency_col
        self._weighted_paths["Noise"] = weighted_paths_noise_col
        self._weighted_paths["SNR"] = weighted_paths_snr_col
        self._route_space["Path"] = weighted_paths_path_col
        # Replace the numerical indexes of the weighted paths with the string indexes of the paths (without the "->")
        self._weighted_paths.index = df_indexes
        # Initialize the routing space as all available
        # For every available channel
        for col_num in range(param.NUMBER_OF_CHANNELS):
            # give it name CH + number of channel and set the respective column to available for every path
            self._route_space["CH" + str(col_num)] = [1] * len(weighted_paths_path_col)
        # Replace the numerical indexes  of the route space with the string indexes of the paths (without the "->")
        self._route_space.index = df_indexes

    # Getters and Setters
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

    @property
    def default_switching_matrix_dict(self):
        return self._default_switching_matrix_dict

    @default_switching_matrix_dict.setter
    def default_switching_matrix_dict(self, default_switching_matrix_dict):
        self._default_switching_matrix_dict = default_switching_matrix_dict

    # Method that links the Nodes and the Lines of the Network setting the successive attributes of the objects.
    # Also applies the switching matrix to the nodes
    def connect(self):
        # For each node of the Network
        for node_name in self.nodes:
            # If node doesn't have a given switching matrix
            if node_name not in self.default_switching_matrix_dict.keys():
                # Create the dictionary for later
                self.nodes[node_name].switching_matrix = {}
                # Set the flag to 1, so that later the switching matrix will be created
                set_default = 1
            # If node has a given switching matrix
            else:
                # Apply the given switching matrix
                self.nodes[node_name].switching_matrix = \
                    copy.deepcopy(self.default_switching_matrix_dict[node_name])
                # The copy is to make a real copy (deep copy) and not a reference, otherwise the default one is updated
                # Set the flag to 1, non need to create a switching matrix
                set_default = 0
            # For each connected node of the current node
            for connected_node in self.nodes[node_name].connected_nodes:
                # Get the line name
                line_name = node_name + connected_node
                # Link the line successive node
                self.lines[line_name].successive[connected_node] = self.nodes[connected_node]
                # Link the node successive line
                self.nodes[node_name].successive[line_name] = self.lines[line_name]
                # If the flag is set (no switching matrix was given)
                if set_default == 1:
                    # Create a dictionary for the connected node within the current node switching matrix dictionary
                    self.nodes[node_name].switching_matrix[connected_node] = {}
                    # Repeat the same operation for the default switching matrix used to restore the actual one
                    self.default_switching_matrix_dict[node_name][connected_node] = {}
                    # For each other connected node of the current node
                    for connected_node_2 in self.nodes[node_name].connected_nodes:
                        # If not the two connected nodes are not the same node
                        if connected_node != connected_node_2:
                            # All channels available between the two nodes
                            self.nodes[node_name].switching_matrix[connected_node][connected_node_2] = \
                                [1] * param.NUMBER_OF_CHANNELS
                            # Repeat the same operation for the default switching matrix used to restore the actual one
                            self.default_switching_matrix_dict[node_name][connected_node][connected_node_2] = \
                                [1] * param.NUMBER_OF_CHANNELS
                        # If the two connected nodes are the same
                        else:
                            # No "U turn" switching from a node to the same node
                            # Set all channels as unavailable
                            self.nodes[node_name].switching_matrix[connected_node][connected_node_2] = \
                                [0] * param.NUMBER_OF_CHANNELS
                            # Repeat the same operation for the default switching matrix used to restore the actual one
                            self.default_switching_matrix_dict[node_name][connected_node][connected_node_2] = \
                                [0] * param.NUMBER_OF_CHANNELS

    # Method used to find the paths between two nodes
    def find_paths(self, label_node1, label_node2):
        # Dictionary used to save the traversal of the nodes, level zero is the starting node
        paths_dict = {0: label_node1}
        # Variable used to check if a new path candidate was found. Now set to one to enter the loop
        new_paths = 1
        # Variable that counts the reached depth of the graph traversal (number of lines that compose the path)
        level = 0
        # As long as new path candidates are discovered
        while new_paths != 0:
            # Reset the new path candidates count
            new_paths = 0
            # Create the vector in the dictionary for the new level, will contain all the paths up to that level
            paths_dict[level + 1] = []
            # For each path at the previous depth level
            for this_level_path in paths_dict[level]:
                # If the last node of the path is not the target destination
                if this_level_path[-1] != label_node2:
                    # Retrieve the lines connected to the last node of the path
                    connected_lines = self.nodes[this_level_path[-1]].successive
                    # For each connected line
                    for connected_line in connected_lines:
                        # If the destination of the line is not already part of the path
                        if connected_line[-1] not in this_level_path:
                            # Create a new path ending with the destination of line of incremented level
                            paths_dict[level + 1].append(this_level_path + connected_line[-1])
                            # If the destination of the new path is not the target destination
                            if connected_line[-1] != label_node2:
                                # Increase the counter of new paths candidates
                                # A path ending with the target destination will not generate new paths
                                new_paths += 1
            # After every path of the level has been checked for longer paths, increase the level
            level += 1
        # Create the list for the collection of the valid paths that connect the requested start and end nodes
        paths = []
        # For every discovered level
        for i_level in range(level):
            # for every discovered path
            for final_path in paths_dict[i_level + 1]:
                # If the final node of the path is the target destination
                if final_path[-1] == label_node2:
                    # Add the path to the list of valid nodes
                    paths.append(final_path)
        # Return the valid paths
        return paths

    # Method for the propagation of the signal information / lightpath. Calls the propagate method of the start node.
    def propagate(self, signal_information):
        prop_signal_information = self.nodes[signal_information.path[0]].propagate(signal_information)
        return prop_signal_information

    # Method that graphically draws the network topology (nodes and lines) in scale
    def draw(self):
        # For each node
        for node_name in self.nodes:
            # Get the coordinates of the node
            x1 = self.nodes[node_name].position[0]
            y1 = self.nodes[node_name].position[1]
            # Plot a circle at the coordinates of the node
            plt.plot(x1, y1, 'go', markersize=5)
            # Add the node name near the nodes circle
            plt.text(x1, y1 + 12000, node_name)
            # For every connected node
            for connected_node_name in self.nodes[node_name].connected_nodes:
                # Get the coordinates of the connected node
                x2 = self.nodes[connected_node_name].position[0]
                y2 = self.nodes[connected_node_name].position[1]
                # Plot a line that connects the two nodes
                plt.plot([x1, x2], [y1, y2], 'b')
        # Add the title
        plt.title('Network graph')
        # Show the graph
        plt.show()

    # Method that searched in the weighted paths dataframe the path that connects two nodes with the best snr
    def find_best_snr(self, label_start_node, label_end_node):
        # Retrieve all the paths that connect two nodes
        paths = self.find_paths(label_start_node, label_end_node)
        # Set the initial best path as an empty string so that if no path is found we pass that information
        best_snr_path = ""
        # Set the initial best SNR as - infinite (the minimum possible in dB)
        best_snr = float('-inf')
        # For a path in the valid paths
        for path in paths:
            # Get the SNR of the path from the weighted paths dataframe
            test_snr = \
                self.weighted_paths.loc[self.weighted_paths["Path"] == util.path_add_arrows(path)]["SNR"].tolist()[0]
            # If the SNR of this path is better then the SNR of the best path and there
            # is a channel available in the path, replace the best path with this path
            if (test_snr > best_snr) and (1 in self.route_space.loc[path].tolist()):
                best_snr_path = path
                best_snr = test_snr
        # Return the best path
        return best_snr_path

    # Method that searched in the weighted paths dataframe the path that connects two nodes with the best latency
    def find_best_latency(self, label_start_node, label_end_node):
        # Retrieve all the paths that connect two nodes
        paths = self.find_paths(label_start_node, label_end_node)
        # Set the initial best path as an empty string so that if no path is found we pass that information
        best_latency_path = ""
        # Set the initial best latency as infinite (the maximum possible)
        best_latency = float('inf')
        # For a path in the valid paths
        for path in paths:
            # Get the SNR of the path from the weighted paths dataframe
            test_latency = self.weighted_paths.loc[self.weighted_paths["Path"] ==
                                                   util.path_add_arrows(path)]["Latency"].tolist()[0]
            # If the latency of this path is better then the latency of the best path and there
            # is a channel available in the path, replace the best path with this path
            if (test_latency < best_latency) and (1 in self.route_space.loc[path].tolist()):
                best_latency_path = path
                best_latency = test_latency
        # Return the best path
        return best_latency_path

    # Method that given a list of connection, streams them and creates
    # such connections assigning to them the best available path
    # Pref is the preference between SNR and Latency, keep_network_once_done if False resets the
    # network to its original state after the streaming, otherwise the occupancies are kept
    def stream(self, stream_connections_list, pref="Latency", keep_network_once_done=False):
        # For each connection to be streamed
        for stream_connection in stream_connections_list:
            # If the preference is Latency, get the path with the best latency that is not occupied
            if pref == "Latency":
                path = self.find_best_latency(stream_connection.input, stream_connection.output)
            # else get the path with the best SNR that is not occupied
            else:
                path = self.find_best_snr(stream_connection.input, stream_connection.output)
            # If a path is found
            if path != "":
                # Find the first available channel
                first_available_channel = self.route_space.loc[path].tolist().index(1) - 1
                # Create a lightpath on the selected path on the first available channel
                lightpath = Lightpath(param.default_input_power, path, "CH" + str(first_available_channel))
                # Calculate the bit rate of this lightpath
                bit_rate = self.calculate_bit_rate(lightpath, self.nodes[path[0]].transceiver)
                # Assign the bit rate to the connection
                stream_connection.bit_rate = bit_rate
                # If the deployable bit rate is >0 the connection is deployed
                if bit_rate > 0:
                    # The lightpath is propagated
                    lightpath = self.propagate(lightpath)
                    # The occupied channel is set as occupied for the path
                    self.route_space.loc[path, "CH" + str(first_available_channel)] = 0
                    # For any path in the route space
                    for path_route in self.route_space.index.tolist():
                        # Get the current channel occupancy of the path
                        occupancy = np.array(self.route_space.loc[path_route].to_list()[1:])
                        # Update the occupation with the switching matrices of all the nodes
                        # on the path that were updated during the propagation of the new lightpath
                        for i in range(len(path_route) - 2):
                            occupancy = \
                                occupancy * np.array(
                                    self.nodes[path_route[i + 1]].switching_matrix[path_route[i]][path_route[i + 2]])
                        # Update the occupation with line occupations of all the lines on the
                        # path that were updated during the propagation of the new lightpath
                        for i in range(len(path_route) - 1):
                            occupancy = occupancy * np.array(self.lines[path_route[i:i + 2]].state)
                        # Update the route space by setting the updated channel of occupancy of the path
                        for channel in range(len(occupancy)):
                            self.route_space.loc[path_route, "CH" + str(channel)] = occupancy[channel]
                    # Set the latency and the SNR of the deployed lightpath
                    stream_connection.latency = lightpath.latency
                    stream_connection.snr = 1 / lightpath.ISNR
                # If the connection cannot be established due to too low SNR with the given transceiver (bit rate = 0),
                # set the latency and SNR to the rejected connection values
                else:
                    stream_connection.latency = 0
                    stream_connection.snr = "None"
            # If a path is not found, set the latency and SNR to the rejected connection values
            else:
                stream_connection.latency = 0
                stream_connection.snr = "None"
        # Restore the network or not depending on the method parameter
        if not keep_network_once_done:
            self.restore_network()
        return stream_connections_list

    # Method used to calculate the maximum obtainable bit rate for a certain lightpath given the transceiver type
    def calculate_bit_rate(self, lightpath, strategy):
        if strategy == "fixed_rate":
            bit_rate = sci_util.bit_rate_fixed(sci_util.db_to_linear(self.weighted_paths["SNR"][lightpath.path]),
                                               lightpath.Rs)
        elif strategy == "flex_rate":
            bit_rate = sci_util.bit_rate_flex(sci_util.db_to_linear(self.weighted_paths["SNR"][lightpath.path]),
                                              lightpath.Rs)
        elif strategy == "shannon":
            bit_rate = sci_util.bit_rate_shannon(sci_util.db_to_linear(self.weighted_paths["SNR"][lightpath.path]),
                                                 lightpath.Rs)
        else:
            bit_rate = 0  # error
        return bit_rate

    # Method that manages the traffic requested with a traffic matrix
    def manage_traffic(self, traffic_matrix):
        # Reset the network to the virgin state
        self.restore_network()
        # Create the list of the connections that have to be made
        connections_to_be_made = []
        # For each starting node of the traffic matrix
        for node_x in traffic_matrix.keys():
            # For each ending node of the connection starting from the other node
            for node_y in traffic_matrix[node_x].keys():
                # If these two nodes are not the same node
                if node_x != node_y and traffic_matrix[node_x][node_y] > 0:
                    # Create a new connection to be made item in the list
                    connections_to_be_made.append((node_x, node_y))
        # Create a list of of all the connections made
        connections_list = []
        # While there are still connections to be made
        while connections_to_be_made:
            # Select randomly a connection to be made
            conn = util.sample_nodes(connections_to_be_made, 1)
            # Get the start and end nodes
            start_node = conn[0][0]
            end_node = conn[0][1]
            # Create a new connection with these start and end nodes
            new_connection = Connection(start_node, end_node, param.default_input_power)
            # Stream the connection without restoring the network
            new_connection = self.stream([new_connection], "SNR", True)
            # Add the streamed connection tot he list of streamed connections
            connections_list += new_connection
            # If this newly streamed connection was not rejected
            if new_connection[0].latency != 0.0 and new_connection[0].snr != "None" \
                    and new_connection[0].bit_rate != 0.0:
                # Subtract the deployed capacity from the traffic matrix
                traffic_matrix[start_node][end_node] -= new_connection[0].bit_rate
                # If no more capacity is needed for this particular connection
                if traffic_matrix[start_node][end_node] <= 0:
                    # Remove the start_node, end_node tuple from the list of connections that have to be made
                    connections_to_be_made.remove((start_node, end_node))
            # If the connection was rejected
            else:
                # Remove the start_node, end_node tuple from the list of connections that have to
                # be made as no capacity can be deployed between those two nodes
                connections_to_be_made.remove((start_node, end_node))
        # Return the list of deployed connections and the post-deployment traffic matrix
        return [connections_list, traffic_matrix]

    # Method the restores the network to a virgin state where no channel si occupied on any line
    def restore_network(self):
        # Restore the nodes switching matrix with the default one
        for node in self.nodes:
            self.nodes[node].switching_matrix = dict(self.default_switching_matrix_dict[node])
        # Restore the channel availability of the lines
        for line in self.lines:
            self.lines[line].state = [1] * param.NUMBER_OF_CHANNELS
        # Restore the route space dataframe to all available
        for col_num in range(param.NUMBER_OF_CHANNELS):
            self.route_space["CH" + str(col_num)] = [1] * len(self.route_space.index)


# Class that models a required connection between two nodes
class Connection:
    # Constructor
    def __init__(self, input, output, signal_power):
        # Input node
        self._input = input
        # Output node
        self._output = output
        # Power of the connection signal
        self._signal_power = signal_power
        # Latency of the connection signal
        self._latency = 0.0
        # SNR of the connection signal
        self._snr = 0.0
        # Bit rate of the connection signal
        self._bit_rate = 0.0

    # Getters and Setters
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

    @property
    def bit_rate(self):
        return self._bit_rate

    @latency.setter
    def latency(self, latency):
        self._latency = latency

    @snr.setter
    def snr(self, snr):
        self._snr = snr

    @bit_rate.setter
    def bit_rate(self, bit_rate):
        self._bit_rate = bit_rate
