import numpy as np
import math
import scipy.io.wavfile
from localize import HyperbolicAsymptote
from localize import getDirectionGivenAsymptotesNew
from tdoa import compute_delay_a_to_b_seconds

# Sample integration
def SampleIntergationLoadSoundFile():
    rate, a = scipy.io.wavfile.read("./Sounds/clap  mic1.wav")
    rate, b = scipy.io.wavfile.read("./Sounds/clap  mic2.wav")
    rate, c = scipy.io.wavfile.read("./Sounds/clap  mic3.wav")
    rate, d = scipy.io.wavfile.read("./Sounds/clap  mic4.wav")

    # Cast to float
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    c = c.astype(np.float64)
    d = d.astype(np.float64)

    # Trim all (since they are 30 second long wav files)
    # to the first 1 seconds.
    trim_length = 1 * rate
    a = a[:trim_length]
    b = b[:trim_length]
    c = c[:trim_length]
    d = d[:trim_length]

    # Compute correlations
    a_b = compute_delay_a_to_b_seconds(a,b, rate)
    a_c = compute_delay_a_to_b_seconds(a,c, rate)
    a_d = compute_delay_a_to_b_seconds(a,d, rate)
    b_c = compute_delay_a_to_b_seconds(b,c, rate)
    b_d = compute_delay_a_to_b_seconds(b,d, rate)
    c_d = compute_delay_a_to_b_seconds(c,d, rate)

    # Geometry of the glasslab setup (update this if different setups are used).
    L = 0.15
    diag_dist = math.sqrt(2 * ((2 * L) ** 2))

    # Create hyperbolae
    hyperbolae = [
        HyperbolicAsymptote.generate_horizontal(2 * L, L, -a_b),
        HyperbolicAsymptote.generate_vertical(2 * L, -L, -a_c),
        HyperbolicAsymptote.generate_diagonal_TL_BR(diag_dist, -a_d),
        HyperbolicAsymptote.generate_diagonal_BL_TR(diag_dist, b_c),
        HyperbolicAsymptote.generate_vertical(2 * L, L, -b_d),
        HyperbolicAsymptote.generate_horizontal(2 * L, -L, -c_d)
    ]

    # Predict angle
    predicted_angle = getDirectionGivenAsymptotesNew(hyperbolae)
    print("Predicted Angle: " + str(predicted_angle))

SampleIntergationLoadSoundFile()