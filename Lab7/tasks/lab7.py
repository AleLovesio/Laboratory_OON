from core import elements as elem
from core import parameters as param
from core import science_utils as sci_util
from core import utils as util

from matplotlib import pyplot as plt
from pathlib import Path


if __name__ == "__main__":
    root = Path(__file__).parent.parent
    types = ["Fixed Rate", "Flexible Rate", "Theoretical Shannon Rate"]
    file = [root / "resources" / "nodes_full_fixed_rate.json", root / "resources" / "nodes_full_flex_rate.json",
            root / "resources" / "nodes_full_shannon.json"]
    network = elem.Network(file[0])
    network_nodes_list = list(network.nodes.keys())
    node_couples = []
    for i in range(100):
        node_couples.append(util.sample_2_nodes(network_nodes_list))
    for j in range(len(types)):
        connections_list = []
        network = elem.Network(file[j])
        for i in range(100):
            connections_list.append(elem.Connection(node_couples[i][0], node_couples[i][1], 1))
        connections_list = network.stream(connections_list, "SNR")
        streams_list = []
        reject_count = 0
        total_capacity = 0
        for connection in connections_list:
            if connection.latency != 0.0 and connection.snr != "None" and connection.bit_rate != 0.0:
                streams_list.append(connection.bit_rate/1e9)
                total_capacity += connection.bit_rate
            else:
                reject_count += 1
        mean_br = total_capacity / (len(streams_list) * 1e9)
        total_capacity = total_capacity / 1e9
        print(types[j] + ": Total deployed capacity: " + str(total_capacity)
              + "Gbps, Average bit rate: " + str(mean_br) + " Gbps, Rejected connections: "+str(100-len(streams_list)))
        plt.figure()
        plt.hist(streams_list, bins=15)
        plt.xlabel("Bit rate [Gbps]")
        plt.ylabel("Paths")
        plt.title("Path choice: SNR, Transceiver type: " + types[j])
        plt.show()
