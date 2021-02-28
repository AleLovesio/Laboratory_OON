import scipy.constants as const

# Physical constants
pi = const.pi
c = const.speed_of_light
h_plank = const.h
# Number of channels per line
NUMBER_OF_CHANNELS = 10
# If true only one very 2  channels can be occupied
side_channel_occupancy = True
# Max Bit error rate
BERt = 1e-3  # 2.5
# Symbol rate
Rs = 32 * 1e9  # s-1
# Noise bandwidth
Bn = 12.5 * 1e9  # Hz
# Max distance between amplifiers/ max span length of a line
MAX_DISTANCE_BETWEEN_AMPLIFIERS = 80000  # m
# Line amplifiers gain
LINE_AMPLIFIER_GAIN = 39.810717  # linear
# Noise figure of the line amplifiers
LINE_AMPLIFIER_NF = 1.995262  # linear
# Center frequency of the C band
C_BAND_CENTER_FREQ = 193.414e12
# default line loss
alpha_default = 23.025851e-6  # 1/m
# Default dispersion
beta_2_default = 2.13e-26  # ps^2/m
# Default nonlinearity coefficient
gamma_default = 1.27e-3  # W^-1 m^-1
# Default frequency spacing between channels
delta_f = 50e9  # Hz
L_eff = 1 / (2 * alpha_default)  # m
# C band:
# 191.2 THz, 195.6 THz
# Default not optimal line launch power
default_input_power = 0.001

