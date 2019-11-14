import numpy as np
import tensorflow as tf
import scipy.io.wavfile
import scipy.misc
import math
import os
import re
import matplotlib.pyplot as plt

def FrameToFeatures(frame_time_domain):
    frame_length = len(frame_time_domain)
    frame_time_domain = np.array(frame_time_domain).astype(np.float32)
    # Accepts a rank 1 np.array [frame_data]

    log_frame_energy = math.log(np.sum(np.square(frame_time_domain)))

    zero_crossing_rate = 0.0
    for i in range(frame_length - 1):
        if (frame_time_domain[i] >= 0) != (frame_time_domain[i + 1] >= 0):
            zero_crossing_rate += 1.0


    signal_delayed = frame_time_domain[1:]
    signal_clipped = frame_time_domain[:-1]

    normalized_autocorrelation_lag_1 = np.dot(signal_clipped, signal_delayed) / math.sqrt(np.sum(np.square(signal_clipped)) * np.sum(np.square(signal_delayed)))

    poly_degree = 12

    predictor_coeffs = np.polyfit(range(frame_length), frame_time_domain, poly_degree)
    first_linear_predictor_coeff = predictor_coeffs[0]

    # log of MSE, had to guess MSE
    predictor_err = np.sum(np.square(np.polyval(predictor_coeffs, range(frame_length)) - frame_time_domain))

    log_linear_predictor_err = math.log(predictor_err)

    return [log_frame_energy, zero_crossing_rate, normalized_autocorrelation_lag_1, first_linear_predictor_coeff, log_linear_predictor_err]

features = FrameToFeatures([1,2,3,4,3,2,3,3,-1,2,-1,-1,3])

print(features)