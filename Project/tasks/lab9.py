from core import elements as elem
from core import parameters as param
from core import science_utils as sci_util
from core import utils as util
from core.utils import print_with_time as print_with_time
from core.utils import add_time as add_time

from matplotlib import pyplot as plt
from pathlib import Path


if __name__ == "__main__":
    Mmax = 10
    enable_histograms = True
    enable_plots_save = True
    with_traffic_matrix = False
    create_log_file = True
    print_log_data_on_console = True
    root = Path(__file__).parent.parent
    types = ["Fixed Rate", "Flexible Rate", "Theoretical Shannon Rate"]
    file = [root / "resources" / "nodes_full_fixed_rate.json", root / "resources" / "nodes_full_flex_rate.json",
            root / "resources" / "nodes_full_shannon.json"]
    save_file_folder = root / "results" / "Lab10"
    network = elem.Network(file[0])
    network_nodes_list = list(network.nodes.keys())

    if not with_traffic_matrix:
        logger = util.Logger(save_file_folder, "Lab10_log_no_traffic_matrix.txt",
                             print_log_data_on_console, create_log_file)
        node_couples = []
        for i in range(100):
            node_couples.append(util.sample_nodes(network_nodes_list, 2))
        for j in range(len(types)):
            print_with_time("Simulating with " + types[j] + " transceiver type")
            connections_list = []
            if j != 0:
                network = elem.Network(file[j])
            for i in range(100):
                connections_list.append(elem.Connection(node_couples[i][0],
                                                        node_couples[i][1], param.default_input_power))
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
            logger.log_line_with_time(types[j] + ": Total deployed capacity: " + str(total_capacity) +
                                      "Gbps, Average bit rate: " + str(mean_br) + " Gbps, Rejected connections: " +
                                      str(100-len(streams_list)))
            plt.figure()
            plt.hist(streams_list, bins=15)
            plt.xlabel("Bit rate [Gbps]")
            plt.ylabel("Paths")
            plt.title("Path choice: SNR, Transceiver type: " + types[j])
            plt.draw()
            plt.pause(0.1)
            if enable_plots_save:
                plt.savefig(save_file_folder / ("Lab10_no_traffic_matrix_" + types[j] + "_histogram.png"))
    else:
        logger = util.Logger(save_file_folder, "Lab10_log_with_traffic_matrix.txt",
                             print_log_data_on_console, create_log_file)
        M = 0
        for j in range(len(types)):
            print_with_time("Simulating with " + types[j] + " transceiver type")
            if j != 0:
                network = elem.Network(file[j])
            connections_list = []
            streams_list = []
            total_capacity_vector = []
            mean_br_vector = []
            reject_count_vector = []
            lightpath_cap_vector = []
            for M in range(1, Mmax + 1):
                print_with_time("Simulating with uniform traffic matrix with "+str(M*100) +
                                " Gbps target capacity per connection.")
                uniform_traffic_matrix = util.generate_uniform_traffic_matrix(network_nodes_list, M)
                [connections_list, traffic_matrix] = network.manage_traffic(uniform_traffic_matrix)

                reject_count = 0
                total_capacity = 0
                streams_list.append([])

                for connection in connections_list:
                    if connection.latency != 0.0 and connection.snr != "None" and connection.bit_rate != 0.0:
                        streams_list[M-1].append(connection.bit_rate / 1e9)
                        total_capacity += connection.bit_rate
                    else:
                        reject_count += 1
                if len(streams_list[M-1]) > 0:
                    mean_br = total_capacity / (len(streams_list[M-1]) * 1e9)
                else:
                    mean_br = 0
                total_capacity = total_capacity / 1e9
                logger.log_line_with_time(types[j] + ": Target capacity per node to node connection: " + str(M*100) +
                                          "Gbps, Total deployed capacity: " + str(total_capacity) +
                                          "Gbps, Average bit rate: " + str(mean_br) + " Gbps, Rejected connections: " +
                                          str(reject_count))
                lightpath_cap_vector.append(M*100)
                total_capacity_vector.append(total_capacity)
                mean_br_vector.append(mean_br)
                reject_count_vector.append(reject_count)

                if enable_histograms:
                    plt.figure(j * (Mmax + 3) + M)
                    plt.hist(streams_list[M-1], bins=15)
                    plt.xlabel("Bit rate [Gbps]")
                    plt.ylabel("Paths")
                    plt.title("Path choice: SNR, Target line capacity: " + str(M*100) + "Gbps, Tran. type: " + types[j])

                    plt.draw()
                    plt.pause(0.1)
                    if enable_plots_save:
                        plt.savefig(save_file_folder / ("Lab10_" + types[j] + "_" + str(M*100) + "Gbps_histogram.png"))

            plt.figure(j * (Mmax + 3) + M + 1)
            plt.plot(lightpath_cap_vector, total_capacity_vector)
            plt.xlabel("Target lightpaths capacity [Gbps]")
            plt.ylabel("Total Capacity [Gbps]")
            plt.title("Path choice: SNR, Target line capacity: " + str(M * 100) + "Gbps, Tran. type: " + types[j])
            plt.draw()
            plt.pause(0.1)
            if enable_plots_save:
                plt.savefig(save_file_folder / ("Lab10_" + types[j] + "_total_capacity_plot.png"))

            plt.figure(j * (Mmax + 3) + M + 2)
            plt.plot(lightpath_cap_vector, mean_br_vector)
            plt.xlabel("Target lightpaths capacity [Gbps]")
            plt.ylabel("Mean bitrate [Gbps]")
            plt.title("Path choice: SNR, Target line capacity: " + str(M * 100) + "Gbps, Tran. type: " + types[j])
            plt.draw()
            plt.pause(0.1)
            if enable_plots_save:
                plt.savefig(save_file_folder / ("Lab10_" + types[j] + "_mean_bitrate_plot.png"))

            plt.figure(j * (Mmax + 3) + M + 3)
            plt.plot(lightpath_cap_vector, reject_count_vector)
            plt.xlabel("Target lightpaths capacity [Gbps]")
            plt.ylabel("Reject count")
            plt.title("Path choice: SNR, Target line capacity: " + str(M * 100) + "Gbps, Tran. type: " + types[j])
            plt.draw()
            plt.pause(0.1)
            if enable_plots_save:
                plt.savefig(save_file_folder / ("Lab10_" + types[j] + "_reject_count_plot.png"))
            if Mmax > 10 and enable_histograms:
                input(add_time("Histograms will be closed. Press [enter] to continue."))
                for close_var in range(1, Mmax + 1):
                    plt.close(j * (Mmax + 3) + close_var)
    print_with_time("Simulation end.")
    plt.show()
    input(add_time("Press [enter] to finish."))
