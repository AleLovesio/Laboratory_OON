from core import parameters as param
from core import science_utils as sci_util
from core import utils as util

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json as json
import copy


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
        self._Rs = param.Rs
        self._df = param.delta_f
        self._ISNR = 0

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

    def increase_noise_power(self, additional_noise_power):
        self.noise_power = additional_noise_power + self.noise_power

    def increase_latency(self, additional_latency):
        self.latency = additional_latency + self.latency

    def update_path(self):
        self.path = self.path[1:]

    def set_launch_power(self, launch_power):
        self.signal_power = launch_power

    def increase_ISNR(self, additional_ISNR):
        self.ISNR = additional_ISNR + self.ISNR


class Lightpath(SignalInformation):
    def __init__(self, signal_power=None, path=None, channel=None):
        super().__init__(signal_power, path)
        if channel:
            self._channel = channel
        else:
            self._channel = "CH0"
            # self._channel = 193.5e12

    @property
    def channel(self):
        return self._channel


class Node:

    def __init__(self, node_data):
        self._label = node_data['label']
        self._position = node_data['position']
        self._connected_nodes = node_data['connected_nodes']
        self._successive = {}
        self._switching_matrix = None
        if 'transceiver' in node_data.keys():
            self._transceiver = node_data['transceiver']
        else:
            self._transceiver = "fixed_rate"

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


class Line:

    def __init__(self, node_data):
        self._label = node_data['label']
        self._length = node_data['length']
        self._successive = {}
        self._state = [1] * param.NUMBER_OF_CHANNELS  # or [0]
        self._n_span = int(np.ceil(self._length / param.MAX_DISTANCE_BETWEEN_AMPLIFIERS))
        self._n_amplifiers = int(2 + np.floor(self._length / param.MAX_DISTANCE_BETWEEN_AMPLIFIERS))
        if 'gain' in node_data.keys():
            self._gain = node_data['gain']
        else:
            self._gain = param.LINE_AMPLIFIER_GAIN
        if 'nf' in node_data.keys():
            self._noise_figure = node_data['nf']
        else:
            self._noise_figure = param.LINE_AMPLIFIER_NF
        if 'alpha' in node_data.keys():
            self._alpha = node_data['alpha']
        else:
            self._alpha = param.alpha_default
        if 'beta_2' in node_data.keys():
            self._beta_2 = node_data['beta_2']
        else:
            self._beta_2 = param.beta_2_default
        if 'gamma' in node_data.keys():
            self._gamma = node_data['gamma']
        else:
            self._gamma = param.gamma_default
        self._L_eff = 1 / (2 * self._alpha)

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

    def latency_generation(self):
        return (3 * self.length) / (2 * param.c)

    def noise_generation(self, lightpath):
        # return 1e-9 * lightpath.signal_power * self.length
        # print("ASE: " + str(self.ase_generation()) + "  NLI: " + str(self.nli_generation(lightpath)))
        return self.ase_generation() + self.nli_generation(lightpath)

    def propagate(self, lightpath):
        lightpath.increase_noise_power(self.noise_generation(lightpath))
        lightpath.increase_ISNR(1 / sci_util.to_snr(lightpath.signal_power, self.noise_generation(lightpath)))
        lightpath.increase_latency(self.latency_generation())
        self.successive[lightpath.path[0]].propagate(lightpath)
        if type(lightpath) == Lightpath:
            self.state[int(lightpath.channel[-1])] = 0
        return lightpath  # to return once the propagation is finished

    def ase_generation(self):
        return sci_util.ase(self.n_amplifiers, param.C_BAND_CENTER_FREQ, param.Bn, self.noise_figure, self.gain)

    def nli_generation(self, lightpath):
        eta_nli = sci_util.nli_eta_nli(self.beta_2, lightpath.Rs, len(self.state),
                                       lightpath.df, self.gamma, self.alpha, self.L_eff)
        # print(eta_nli)
        return sci_util.nli(lightpath.signal_power, eta_nli, self.n_span, param.Bn)

    def optimized_launch_power(self, lightpath=None):
        if not lightpath:
            lightpath = Lightpath()
        return sci_util.opt_launch_pwr(self.ase_generation(), sci_util.nli_eta_nli(self.beta_2, lightpath.Rs,
                                                                                   len(self.state), lightpath.df,
                                                                                   self.gamma, self.alpha, self.L_eff),
                                       self.n_span, param.Bn)


class Network:

    def __init__(self, json_data_file):
        self._nodes = {}
        self._lines = {}
        self._nodes_data = json.load(open(json_data_file, 'r'))
        self._weighted_paths = pd.DataFrame()
        self._route_space = pd.DataFrame()
        self._default_switching_matrix_dict = {}
        for key in self._nodes_data:
            node_pos = tuple(self._nodes_data[key]["position"])
            conn_nodes = self._nodes_data[key]["connected_nodes"]
            if "transceiver" in self._nodes_data[key].keys():
                transceiver = self._nodes_data[key]["transceiver"]
            else:
                transceiver = "fixed_rate"
            self._nodes[key] = \
                Node({'label': key, 'position': node_pos, 'connected_nodes': conn_nodes, 'transceiver': transceiver})
            if "switching_matrix" in self._nodes_data[key].keys():
                self._default_switching_matrix_dict[key] = self._nodes_data[key]["switching_matrix"]
            for second_node_str in conn_nodes:
                line_name = key + second_node_str
                second_node_pos = self._nodes_data[second_node_str]["position"]
                line_length = sci_util.line_len(node_pos, second_node_pos)
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
                        df_indexes.append(weighted_paths_path)
                        weighted_paths_path_col.append(util.path_add_arrows(weighted_paths_path))
                        weighted_paths_sig_inf = SignalInformation(param.default_input_power, weighted_paths_path)
                        weighted_paths_sig_inf = self.propagate(weighted_paths_sig_inf)
                        weighted_paths_latency_col.append(weighted_paths_sig_inf.latency)
                        weighted_paths_noise_col.append(weighted_paths_sig_inf.noise_power)
                        # weighted_paths_snr_col.append(
                        #    sci_util.to_snr(weighted_paths_sig_inf.signal_power, weighted_paths_sig_inf.noise_power))
                        weighted_paths_snr_col.append(1 / weighted_paths_sig_inf.ISNR)
        self._weighted_paths["Path"] = weighted_paths_path_col
        self._weighted_paths["Latency"] = weighted_paths_latency_col
        self._weighted_paths["Noise"] = weighted_paths_noise_col
        self._weighted_paths["SNR"] = weighted_paths_snr_col
        self._route_space["Path"] = weighted_paths_path_col
        self._weighted_paths.index = df_indexes
        for col_num in range(param.NUMBER_OF_CHANNELS):
            self._route_space["CH" + str(col_num)] = [1] * len(weighted_paths_path_col)
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

    @property
    def default_switching_matrix_dict(self):
        return self._default_switching_matrix_dict

    def connect(self):
        for node_name in self.nodes:
            if node_name not in self.default_switching_matrix_dict.keys():
                self.nodes[node_name].switching_matrix = {}
                set_default = 1
            else:
                self.nodes[node_name].switching_matrix = \
                    copy.deepcopy(self.default_switching_matrix_dict[node_name])
                # copy is to make a real copy (deep copy) and not a reference
                set_default = 0
            for connected_node in self.nodes[node_name].connected_nodes:
                line_name = node_name + connected_node
                self.lines[line_name].successive[connected_node] = self.nodes[connected_node]
                self.nodes[node_name].successive[line_name] = self.lines[line_name]
                if set_default == 1:
                    self.nodes[node_name].switching_matrix[connected_node] = {}
                    self.default_switching_matrix_dict[node_name][connected_node] = {}
                    for connected_node_2 in self.nodes[node_name].connected_nodes:
                        if connected_node != connected_node_2:
                            self.nodes[node_name].switching_matrix[connected_node][connected_node_2] = \
                                [1] * param.NUMBER_OF_CHANNELS
                            self.default_switching_matrix_dict[node_name][connected_node][connected_node_2] = \
                                [1] * param.NUMBER_OF_CHANNELS
                        else:
                            self.nodes[node_name].switching_matrix[connected_node][connected_node_2] = \
                                [0] * param.NUMBER_OF_CHANNELS
                            self.default_switching_matrix_dict[node_name][connected_node][connected_node_2] = \
                                [0] * param.NUMBER_OF_CHANNELS

    def find_paths(self, label_node1, label_node2):
        paths_dict = {0: label_node1}
        new_paths = 1
        level = 0
        while new_paths != 0:
            new_paths = 0
            paths_dict[level + 1] = []
            for this_level_path in paths_dict[level]:
                if this_level_path[-1] != label_node2:
                    connected_lines = self.nodes[this_level_path[-1]].successive
                    for connected_line in connected_lines:
                        if connected_line[-1] not in this_level_path:
                            paths_dict[level + 1].append(this_level_path + connected_line[-1])
                            if connected_line[-1] != label_node2:
                                new_paths += 1
            level += 1
        paths = []
        for i_level in range(level):
            for final_path in paths_dict[i_level + 1]:
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
                self.weighted_paths.loc[self.weighted_paths["Path"] == util.path_add_arrows(path)]["SNR"].tolist()[0]
            if (test_snr > best_snr) and (1 in self.route_space.loc[path].tolist()):
                best_snr_path = path
                best_snr = test_snr
        return best_snr_path

    def find_best_latency(self, label_start_node, label_end_node):
        paths = self.find_paths(label_start_node, label_end_node)
        best_latency_path = ""
        best_latency = float('inf')
        for path in paths:
            test_latency = self.weighted_paths.loc[self.weighted_paths["Path"] ==
                                                   util.path_add_arrows(path)]["Latency"].tolist()[0]
            if (test_latency < best_latency) and (1 in self.route_space.loc[path].tolist()):
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
                first_available_channel = self.route_space.loc[path].tolist().index(1) - 1  # find the first one
                lightpath = Lightpath(param.default_input_power, path, "CH" + str(first_available_channel))
                bit_rate = self.calculate_bit_rate(lightpath, self.nodes[path[0]].transceiver)
                stream_connection.bit_rate = bit_rate
                if bit_rate > 0:
                    lightpath = self.propagate(lightpath)
                    self.route_space.loc[path, "CH" + str(first_available_channel)] = 0
                    for path_route in self.route_space.index.tolist():
                        occupancy = np.array(self.route_space.loc[path_route].to_list()[1:])
                        # update occupation with switching matrix
                        for i in range(len(path_route) - 2):
                            occupancy = \
                                occupancy * np.array(
                                    self.nodes[path_route[i + 1]].switching_matrix[path_route[i]][path_route[i + 2]])
                        # update occupation with line occupation
                        for i in range(len(path_route) - 1):
                            occupancy = occupancy * np.array(self.lines[path_route[i:i + 2]].state)
                        for channel in range(len(occupancy)):
                            self.route_space.loc[path_route, "CH" + str(channel)] = occupancy[channel]
                    stream_connection.latency = lightpath.latency
                    # stream_connection.snr = sci_util.to_snr(lightpath.signal_power, lightpath.noise_power)
                    stream_connection.snr = 1 / lightpath.ISNR
                else:
                    stream_connection.latency = 0
                    stream_connection.snr = "None"
            else:
                stream_connection.latency = 0
                stream_connection.snr = "None"
        for node in self.nodes:
            self.nodes[node].switching_matrix = dict(self.default_switching_matrix_dict[node])
        for line in self.lines:
            self.lines[line].state = [1] * param.NUMBER_OF_CHANNELS
        for col_num in range(param.NUMBER_OF_CHANNELS):
            self.route_space["CH" + str(col_num)] = [1] * len(self.route_space.index)
        return stream_connections_list

    def calculate_bit_rate(self, lightpath, strategy):
        if strategy == "fixed_rate":
            bit_rate = sci_util.bit_rate_fixed(self.weighted_paths["SNR"][lightpath.path], lightpath.Rs)
        elif strategy == "flex_rate":
            bit_rate = sci_util.bit_rate_flex(self.weighted_paths["SNR"][lightpath.path], lightpath.Rs)
        elif strategy == "shannon":
            bit_rate = sci_util.bit_rate_shannon(self.weighted_paths["SNR"][lightpath.path], lightpath.Rs)
        else:
            bit_rate = 0  # error
        return bit_rate


class Connection:

    def __init__(self, input, output, signal_power):
        self._input = input
        self._output = output
        self._signal_power = signal_power
        self._latency = 0.0
        self._snr = 0.0
        self._bit_rate = 0.0

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
