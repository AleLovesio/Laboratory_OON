from core import elements as elem
from core import parameters as param
from core import science_utils as sci_util
from core import utils as util

import matplotlib.pyplot as plt
from pathlib import Path

if __name__ == "__main__":
    snr_analysis_only = False
    root = Path(__file__).parent.parent
    file = root / "resources" / "nodes.json"
    network = elem.Network(file)
    network_nodes_list = list(network.nodes.keys())
    connections_list = []
    for i in range(100):
        [start_node, end_node] = util.sample_nodes(network_nodes_list, 2)
        connections_list.append(elem.Connection(start_node, end_node, 1))
    if snr_analysis_only:
        analyses = ["SNR"]
    else:
        analyses = ["Latency", "SNR"]
    for analysis_type in analyses:
        connections_list = network.stream(connections_list, analysis_type)
        streams_snr_list = []
        streams_latency_list = []
        for connection in connections_list:
            streams_snr_list.append(connection.snr)
            streams_latency_list.append(sci_util.linear_to_db(connection.latency))
        plt.figure()
        if analysis_type == "Latency":
            plt.hist(streams_latency_list, bins=15)
            unit = "[s]"
        else:
            plt.hist(streams_snr_list, bins=15)
            unit = "[dB]"
        plt.xlabel(analysis_type+" Range "+unit)
        plt.ylabel("Paths")
        plt.title("Path choice: "+analysis_type)
    plt.show()
