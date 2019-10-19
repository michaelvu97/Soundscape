import numpy as np
import math
import scipy.io.wavfile
from gradienttest import generate_horizontal
from gradienttest import generate_vertical
from gradienttest import generate_diagonal_BL_TR
from gradienttest import generate_diagonal_TL_BR
from gradienttest import getDirectionGivenAsymptotesNew
from gradienttest import HyperbolicAsymptote

rate, a = scipy.io.wavfile.read("./clap  mic1.wav")
rate, b = scipy.io.wavfile.read("./clap  mic2.wav")
rate, c = scipy.io.wavfile.read("./clap  mic3.wav")
rate, d = scipy.io.wavfile.read("./clap  mic4.wav")
a = a.astype(np.float64)
b = b.astype(np.float64)
c = c.astype(np.float64)
d = d.astype(np.float64)

length = rate * 2

d_to_a_corr = 579+73
c_delay_corr = d_to_a_corr
a_delay_corr = 524 + 47 + c_delay_corr
b_delay_corr = 403 + a_delay_corr
a = np.array(a[a_delay_corr:a_delay_corr+length])
b = np.array(b[b_delay_corr:length+b_delay_corr])
c = np.array(c[c_delay_corr:c_delay_corr+length])
d = np.array(d[:length])

def compute_delay_a_to_b_seconds(a, b, rate):


    energies = {}

    # Computed by determining the max sample time delay between phyiscal microphones
    dist_meters = 0.3
    max_delay_seconds = dist_meters / 343
    max_delay_samples =math.ceil(max_delay_seconds * rate)
    # delay_range = range(-max_delay_samples, max_delay_samples)
    delay_range = range(-600, 600) # TODO dynamic / correct

    def cross_correlation(a, b, delay_b):
        N = len(a)
        N_window = N - abs(delay_b)
        if N_window <= 0:
            print("ERR: NEGATIVE N")
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
    # print("The sample delay to b is " + str(delay_to_b_samples) + "[negative means b lags] (" + str(delay_to_b_samples / rate) + " seconds)")
    print(delay_to_b_samples)
    return delay_to_b_samples / rate

a_b = compute_delay_a_to_b_seconds(a,b, rate)
a_c = compute_delay_a_to_b_seconds(a,c, rate)
a_d = compute_delay_a_to_b_seconds(a,d, rate)
b_c = compute_delay_a_to_b_seconds(b,c, rate)
b_d = compute_delay_a_to_b_seconds(b,d, rate)
c_d = compute_delay_a_to_b_seconds(c,d, rate)

print("a_b:" + str(a_b))
print("a_c:" + str(a_c))
print("a_d:" + str(a_d))
print("b_c:" + str(b_c))
print("b_d:" + str(b_d))
print("c_d:" + str(c_d))

L = 0.15
diag_dist = math.sqrt(2 * ((2 * L) ** 2))

hyperbolae = [
    generate_horizontal(2 * L, L, -a_b),
    generate_vertical(2 * L, -L, -a_c),
    generate_diagonal_TL_BR(diag_dist, -a_d),
    generate_diagonal_BL_TR(diag_dist, b_c), # TODO verify delay direction
    generate_vertical(2 * L, L, -b_d),
    generate_horizontal(2 * L, -L, -c_d)
]

print("The predicted angle is: " + str(getDirectionGivenAsymptotesNew(hyperbolae)))