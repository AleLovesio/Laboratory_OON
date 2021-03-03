from core import elements as elem
from core import parameters as param
from core import science_utils as sci_util
from core import utils as util

import pandas as pd
from pathlib import Path


if __name__ == "__main__":
    root = Path(__file__).parent.parent
    file = root / "resources" / "nodes.json"
    network = elem.Network(file)
    dataframe = pd.DataFrame()
    path_col = []
    latency_col = []
    noise_col = []
    snr_col = []
    for start_node in network.nodes.keys():
        for end_node in network.nodes.keys():
            if start_node != end_node:
                for path in network.find_paths(start_node, end_node):
                    path_col.append(util.path_add_arrows(path))
                    sig_inf = elem.SignalInformation(1, path)
                    sig_inf = network.propagate(sig_inf)
                    latency_col.append(sig_inf.latency)
                    noise_col.append(sig_inf.noise_power)
                    snr_col.append(sci_util.to_snr(sig_inf.signal_power, sig_inf.noise_power))
    dataframe["Path"] = path_col
    dataframe["Latency"] = latency_col
    dataframe["Noise"] = noise_col
    dataframe["SNR"] = snr_col
    print(dataframe)
    network.draw()
