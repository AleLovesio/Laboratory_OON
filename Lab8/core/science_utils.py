import numpy as np
from core import parameters as param


# function that returns the SNR
def to_snr(signal_power, noise_power):
    return 10 * np.log10(signal_power / noise_power)


# function that returns the distance between two points
def line_len(node_pos, second_node_pos):
    return np.sqrt((node_pos[0] - second_node_pos[0]) ** 2 + (node_pos[1] - second_node_pos[1]) ** 2)


# function to calculate the bit rate for fixed rate strategy
def bit_rate_fixed(gsnr):
    if gsnr >= (4 * param.BERt * param.Rs / param.Bn):
        bit_rate = 1e11  # 100Gbps, PM-QPSK
    else:
        bit_rate = 0  # 0Gbps
    return bit_rate


# function to calculate the bit rate for flex rate strategy
def bit_rate_flex(gsnr):
    if gsnr >= ((80/3) * param.BERt * param.Rs / param.Bn):
        bit_rate = 4e11  # 400Gbps, PM-16QAM
    elif gsnr >= (7 * param.BERt * param.Rs / param.Bn):
        bit_rate = 2e11  # 200Gbps, PM-8QAM
    elif gsnr >= (4 * param.BERt * param.Rs / param.Bn):
        bit_rate = 1e11  # 100Gbps, PM-QPSK
    else:
        bit_rate = 0  # 0Gbps
    return bit_rate


# function to calculate the bit rate for shannon strategy
def bit_rate_shannon(gsnr):
    # theoretical Shannon rate with an ideal Gaussian modulation
    bit_rate = 2 * param.Rs * np.log2(1 + (gsnr * param.Bn / param.Rs))
    return bit_rate


# function to calculate the ASE
#      [adimensional]      [Hz]          [Hz]     [linear] [linear]
def ase(n_amplifiers, freq_band_center, noise_bw, noise_fig, gain):
    return n_amplifiers * param.h_plank * freq_band_center * noise_bw * noise_fig * (gain - 1)


# function to calculate the nonlinear interference.
def nli(Pch, eta_nli, N_span):
    return (Pch**3) * eta_nli * N_span


# function to calculate eta_nli for nonlinear interference
def nli_eta_nli(beta_2, Rs, Nch, delta_f, gamma, alpha):
    return (16 / (17 * param.pi)) * \
           np.log(((param.pi ** 2) / 2) * (beta_2 * (Rs ** 2) / alpha) * (Nch ** (2 * Rs / delta_f))) * \
           ((gamma ** 2)/(4 * alpha * beta_2)) * (1/(Rs ** 3))
