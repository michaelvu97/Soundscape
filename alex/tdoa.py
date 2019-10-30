import numpy as np
import math
import scipy.io.wavfile

def compute_delay_a_to_b_seconds(a, b, rate):

    energies = {}

    # Computed by determining the max sample time delay between phyiscal microphones
    dist_meters = 0.3
    max_delay_seconds = dist_meters / 343
    max_delay_samples =math.ceil(max_delay_seconds * rate)
    delay_range = range(-max_delay_samples, max_delay_samples)
    # delay_range = range(-600, 600) # TODO dynamic / correct

    def cross_correlation(a, b, delay_b):
        N = len(a)
        N_window = N - abs(delay_b)
        if N_window <= 0:
            return 0

        a_start = 0
        b_start = 0
        if delay_b < 0:
            b_start = -delay_b
        else:
            a_start = delay_b

        a_end = a_start + N_window
        b_end = b_start + N_window

        a_window = a[a_start:a_end]
        b_window = b[b_start:b_end]

        a_window_mean = np.mean(a_window)
        b_window_mean = np.mean(b_window)

        a_window_centralized = a_window - a_window_mean
        b_window_centralized = b_window - b_window_mean

        a_window_std_dev = math.sqrt(np.mean(np.square(a_window_centralized)))
        b_window_std_dev = math.sqrt(np.mean(np.square(b_window_centralized)))

        # Cross correlation of the windows.
        return np.mean((a_window_centralized) * (b_window_centralized)) / (a_window_std_dev * b_window_std_dev)

    for i in delay_range:
        corr = cross_correlation(a, b, i)
        # print("delay=" + str(i) + ", corr=" + str(corr))
        energies[i] = corr

    delay_to_b_samples = max(energies, key=energies.get)
    return delay_to_b_samples / rate
