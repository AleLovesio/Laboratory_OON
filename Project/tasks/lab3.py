from core import elements as elem
from core import parameters as param
from core import science_utils as sci_util
from core import utils as util

import matplotlib.pyplot as plt
from pathlib import Path

if __name__ == "__main__":
    root = Path(__file__).parent.parent
    file = root / "resources" / "nodes.json"
    network = elem.Network(file)
    network_nodes_list = list(network.nodes.keys())
    connections_list = []
    for i in range(100):
        [start_node, end_node] = util.sample_nodes(network_nodes_list, 2)
        connections_list.append(elem.Connection(start_node, end_node, 1))
    for analysis_type in ["Latency", "SNR"]:
        connections_list = network.stream(connections_list, analysis_type)
        streams_snr_list = []
        streams_latency_list = []
        for connection in connections_list:
            streams_snr_list.append(connection.snr)
            streams_latency_list.append(connection.latency)
        plt.figure()
        plt.hist(streams_snr_list, bins=15)
        if analysis_type == "Latency":
            unit = "[s]"
        else:
            unit = "[dB]"
        plt.xlabel(analysis_type+" Range "+unit)
        plt.ylabel("Paths")
        plt.title("Path choice: "+analysis_type)
    plt.show()
