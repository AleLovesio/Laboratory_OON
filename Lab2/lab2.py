import pandas as pd

class Signal_information:
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

    def increase_signal_power(self, additional_signal_power):
        self.signal_power = additional_signal_power + self.signal_power

    def increase_noise_power(self, additional_noise_power):
        self.noise_power = additional_noise_power + self.noise_power

    def increase_latency(self, additional_latency):
        self.latency = additional_latency + self.latency

    def update_path(self):
        self.path = self.path[1:]

class Node:
    def __init__(self, node_data):
        self._label = node_data['label']
        self._position = node_data['position']
        self._connected_nodes = node_data['connected_nodes']
        self._successive = {}

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, label):
        self._label = label

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, position):
        self._position = position

    @property
    def connected_nodes(self):
        return self._connected_nodes

    @connected_nodes.setter
    def connected_nodes(self, connected_nodes):
        self._connected_nodes = connected_nodes

    @property
    def successive(self):
        return self._successive

    @successive.setter
    def successive(self, successive):
        self._successive = successive

    def propagate(self,signal_information):
        signal_information.update_path()
        self.successive[signal_information.path[0]].propagate()


