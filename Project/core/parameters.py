import scipy.constants as const

pi = const.pi
c = const.speed_of_light
NUMBER_OF_CHANNELS = 10
side_channel_occupancy = True
BERt = 1e-3  # 2.5
Rs = 32 * 1e9
Bn = 12.5 * 1e9
MAX_DISTANCE_BETWEEN_AMPLIFIERS = 80000
LINE_AMPLIFIER_GAIN = 39.810717
LINE_AMPLIFIER_NF = 1.995262
h_plank = const.h
C_BAND_CENTER_FREQ = 193.414e12
alpha_default = 23.025851e-6  # 1/m
beta_2_default = 2.13e-26  # ps^2/m
gamma_default = 1.27e-3  # W^-1 m^-1
delta_f = 50e9  # Hz
L_eff = 1 / (2 * alpha_default)  # m
# C band:
# 191.2 THz, 195.6 THz

default_input_power = 0.001

