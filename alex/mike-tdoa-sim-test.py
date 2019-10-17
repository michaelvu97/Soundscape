import numpy as np
import math
import scipy.io.wavfile

rate, a = scipy.io.wavfile.read("./clap  mic1.wav")
rate, b = scipy.io.wavfile.read("./clap  mic2.wav")

length = rate * 2

a = np.array(a[:length])
b = np.array(b[:length])

noise_magnitude = 0.25

# Simulate gaussian noise in the micrphone
noise_a = noise_magnitude * np.random.normal(size=len(a))
noise_b = noise_magnitude * np.random.normal(size=len(a))

# Simulate attenuation
b = b * 0.25

# Add noise
a = a + noise_a
b = b + noise_b




# Delays 
def delay(arr, tau):
    if tau == 0:
        return arr
    
    if tau > 0:
        return np.concatenate((np.zeros(tau), arr[:-tau]))

    # Negative delay
    test = (arr[-tau:], np.zeros(-tau))
    return np.concatenate(test)

# print(np.flip(np.correlate(a,b, mode='same'), axis=0))
# print(np.correlate(a,b, mode='valid'))

energies = {}

# Computed by determining the max sample time delay between phyiscal microphones
dist_meters = 0.3
max_delay_seconds = dist_meters / 343
max_delay_samples =math.ceil(max_delay_seconds * rate)
delay_range = range(-max_delay_samples, max_delay_samples)

for i in delay_range:
    # print("i:" + str(i) + ", " + str(delay(b,i)))
    corr = np.correlate(a, delay(b, i), mode='valid')
    energy = np.sum(corr ** 2)
    print("delay=" + str(i) + ", corr=" + str(corr) + ", energy=" + str(energy))
    energies[i] = energy

delay_to_b_samples = max(energies, key=energies.get)
print("The sample delay to b is " + str(delay_to_b_samples) + "[negative means b lags] (" + str(delay_to_b_samples / rate) + " seconds)")


