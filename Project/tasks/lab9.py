from pathlib import Path

from matplotlib import pyplot as plt

from core import elements as elem
from core import parameters as param
from core import science_utils as sci_util
from core import utils as util
from core.utils import add_time as add_time
from core.utils import print_with_time as print_with_time

# Main program
if __name__ == "__main__":
    # Set maximum requested capacity per line, where the requested capacity per line is M*100Gbps
    M_max = 30
    # Enable the drawing of the histograms
    enable_histograms = True
    # Enable the saving to disk of the plots
    enable_plots_save = False
    # Enable the usage of the traffic matrix (False: first main of Lab10, True: second part)
    with_traffic_matrix = True
    # Log to file the obtained results
    create_log_file = True
    # Print the data that is logged to file also on the console
    print_log_data_on_console = True
    # Ask in the console to close the histograms before changing the transceiver type or close automatically
    ask_to_close = False
    # Project folder
    root = Path(__file__).parent.parent
    # Transceiver types
    types = ["Fixed Rate", "Flexible Rate", "Theoretical Shannon Rate"]
    # Network initialization files
    file = [root / "resources" / "nodes_full_fixed_rate.json", root / "resources" / "nodes_full_flex_rate.json",
            root / "resources" / "nodes_full_shannon.json"]
    # Folder where to save the results
    save_file_folder = root / "results" / "Lab10"
    # Create the first Network in order to get the nodes names
    network = elem.Network(file[0])
    # Get the nodes names
    network_nodes_list = list(network.nodes.keys())
    # First main type of Lab10:
    if not with_traffic_matrix:
        # Create the logger object in order to save the text results
        logger = util.Logger(save_file_folder, "Lab10_log_no_traffic_matrix.txt",
                             print_log_data_on_console, create_log_file)
        # Create list for the nodes couple selection for the connection creation
        node_couples = []
        # Pick 100 couples of nodes
        for i in range(100):
            node_couples.append(util.sample_nodes(network_nodes_list, 2))
        # Repeat the simulations for every transceiver type
        for j in range(len(types)):
            print_with_time("Simulating with " + types[j] + " transceiver type")
            # Create list for the set of connections to be streamed
            connections_list = []
            # Update the Network with another transceiver, if it is not the first time
            if j != 0:
                network = elem.Network(file[j])
            # Create the 100 connections with the 100 node couples and put them in the list
            for i in range(100):
                connections_list.append(elem.Connection(node_couples[i][0],
                                                        node_couples[i][1], param.default_input_power))
            # Stream the connections
            connections_list = network.stream(connections_list, "SNR")
            # List that will contain the bit rates of the connections in Gbps
            streams_list = []
            # Counter of rejected connections
            reject_count = 0
            # Total capacity counter
            total_capacity = 0
            # For every streamed connection
            for connection in connections_list:
                # If the connection wasn't rejected
                if connection.latency != 0.0 and connection.snr != "None" and connection.bit_rate != 0.0:
                    # Save to the vector the connection bit rate
                    streams_list.append(connection.bit_rate/1e9)
                    # Increase the total capacity by the connection deployed capacity
                    total_capacity += connection.bit_rate
                # If the connection was rejected
                else:
                    # Increase the rejected connection counter
                    reject_count += 1
            # Calculate the average bitrate
            mean_br = total_capacity / (len(streams_list) * 1e9)
            # Calculate in Gbps the total deployed capacity
            total_capacity = total_capacity / 1e9
            # Log the calculated results
            logger.log_line_with_time(types[j] + ": Total deployed capacity: " + str(total_capacity) +
                                      "Gbps, Average bit rate: " + str(mean_br) + " Gbps, Rejected connections: " +
                                      str(100-len(streams_list)))
            # Create an histogram of the bitrate of the deployed connections
            plt.figure()
            plt.hist(streams_list, bins=15)
            plt.xlabel("Bit rate [Gbps]")
            plt.ylabel("Paths")
            plt.title("Path choice: SNR, Transceiver type: " + types[j])
            plt.draw()
            plt.pause(0.1)
            # Save the figure if enabled
            if enable_plots_save:
                plt.savefig(save_file_folder / ("Lab10_no_traffic_matrix_" + types[j] + "_histogram.png"))
    # Second main type of Lab10:
    else:
        # Create the logger object in order to save the text results
        logger = util.Logger(save_file_folder, "Lab10_log_with_traffic_matrix.txt",
                             print_log_data_on_console, create_log_file)
        # Initialize the requested capacity per line, where the requested capacity per line is M*100Gbps
        M = 0
        # Repeat the simulations for every transceiver type
        for j in range(len(types)):
            print_with_time("Simulating with " + types[j] + " transceiver type")
            # Update the Network with another transceiver, if it is not the first time
            if j != 0:
                network = elem.Network(file[j])
            # Initialize all the lists that will save the results of the simulation
            connections_list = []
            streams_list = []
            total_capacity_vector = []
            mean_br_vector = []
            reject_count_vector = []
            lightpath_cap_vector = []
            # For every requested capacity per line
            for M in range(1, M_max + 1):
                print_with_time("Simulating with uniform traffic matrix with "+str(M*100) +
                                " Gbps target capacity per connection.")
                # Generate the uniform traffic matrix with M*100Gbps traffic request for every line
                uniform_traffic_matrix = util.generate_uniform_traffic_matrix(network_nodes_list, M)
                # Deploy the traffic
                [connections_list, traffic_matrix] = network.manage_traffic(uniform_traffic_matrix)
                # Initialize the counters
                # Counter of rejected connections of the current M
                reject_count = 0
                # Total capacity counter
                total_capacity = 0
                # Add a list that will contain the bit rates of the connections in Gbps for every M
                streams_list.append([])
                # For every streamed connection
                for connection in connections_list:
                    # If the connection wasn't rejected
                    if connection.latency != 0.0 and connection.snr != "None" and connection.bit_rate != 0.0:
                        # Save to the vector the connection bit rate
                        streams_list[M-1].append(connection.bit_rate / 1e9)
                        # Increase the total capacity by the connection deployed capacity
                        total_capacity += connection.bit_rate
                    # If the connection was rejected
                    else:
                        # Increase the rejected connection counter
                        reject_count += 1
                # Calculate the average bitrate
                if len(streams_list[M-1]) > 0:
                    mean_br = total_capacity / (len(streams_list[M-1]) * 1e9)
                else:
                    mean_br = 0
                # Calculate in Gbps the total deployed capacity
                total_capacity = total_capacity / 1e9
                # Log the calculated results
                logger.log_line_with_time(types[j] + ": Target capacity per node to node connection: " + str(M*100) +
                                          "Gbps, Total deployed capacity: " + str(total_capacity) +
                                          "Gbps, Average bit rate: " + str(mean_br) + " Gbps, Rejected connections: " +
                                          str(reject_count))
                # Save the data that was logged to make the final plots after all M capacities are simulated
                lightpath_cap_vector.append(M*100)
                total_capacity_vector.append(total_capacity)
                mean_br_vector.append(mean_br)
                reject_count_vector.append(reject_count)
                # Create an histogram of the bitrate of the deployed connections if enabled
                if enable_histograms:
                    plt.figure(j * (M_max + 3) + M)
                    plt.hist(streams_list[M-1], bins=15)
                    plt.xlabel("Bit rate [Gbps]")
                    plt.ylabel("Paths")
                    plt.title("Path choice: SNR, Target line capacity: " + str(M*100) + "Gbps, Tran. type: " + types[j])
                    plt.draw()
                    plt.pause(0.1)
                    # Save the figure if enabled
                    if enable_plots_save:
                        plt.savefig(save_file_folder / ("Lab10_" + types[j] + "_" + str(M*100) + "Gbps_histogram.png"))
            # Plot the total deployed capacity as function of M, the requested capacity
            plt.figure(j * (M_max + 3) + M + 1)
            plt.plot(lightpath_cap_vector, total_capacity_vector)
            plt.xlabel("Target lightpaths capacity [Gbps]")
            plt.ylabel("Total Capacity [Gbps]")
            plt.title("Path choice: SNR, Target line capacity: " + str(M * 100) + "Gbps, Tran. type: " + types[j])
            plt.draw()
            plt.pause(0.1)
            # Save the figure if enabled
            if enable_plots_save:
                plt.savefig(save_file_folder / ("Lab10_" + types[j] + "_total_capacity_plot.png"))
            # Plot the average bitrate as function of M, the requested capacity
            plt.figure(j * (M_max + 3) + M + 2)
            plt.plot(lightpath_cap_vector, mean_br_vector)
            plt.xlabel("Target lightpaths capacity [Gbps]")
            plt.ylabel("Mean bitrate [Gbps]")
            plt.title("Path choice: SNR, Target line capacity: " + str(M * 100) + "Gbps, Tran. type: " + types[j])
            plt.draw()
            plt.pause(0.1)
            # Save the figure if enabled
            if enable_plots_save:
                plt.savefig(save_file_folder / ("Lab10_" + types[j] + "_mean_bitrate_plot.png"))
            # Plot the reject count as function of M, the requested capacity
            plt.figure(j * (M_max + 3) + M + 3)
            plt.plot(lightpath_cap_vector, reject_count_vector)
            plt.xlabel("Target lightpaths capacity [Gbps]")
            plt.ylabel("Reject count")
            plt.title("Path choice: SNR, Target line capacity: " + str(M * 100) + "Gbps, Tran. type: " + types[j])
            plt.draw()
            plt.pause(0.1)
            # Save the figure if enabled
            if enable_plots_save:
                plt.savefig(save_file_folder / ("Lab10_" + types[j] + "_reject_count_plot.png"))
            # If there are too many histogram we must close them
            if M_max > 10 and enable_histograms:
                # Halt the execution and ask to continue
                if ask_to_close:
                    input(add_time("Histograms will be closed. Press [enter] to continue."))
                # Close them
                for close_var in range(1, M_max + 1):
                    plt.close(j * (M_max + 3) + close_var)
    print_with_time("Simulation end.")
    # SHow the plots and halt the execution
    plt.show()
    # Quit
    input(add_time("Press [enter] to finish."))
