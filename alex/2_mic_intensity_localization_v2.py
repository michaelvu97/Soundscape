#    !/Library/Frameworks/Python.framework/Versions/3.6/bin/python3
import numpy as np
import math
import scipy.io.wavfile



rate, a = scipy.io.wavfile.read("./Honk Trimmed mic 3.wav")
rate, b = scipy.io.wavfile.read("./Honk Trimmed mic 4.wav")

#rate, a = scipy.io.wavfile.read("./honk mic 3.wav")
#rate, b = scipy.io.wavfile.read("./honk mic 4.wav")

a = a.astype(np.float64)
b = b.astype(np.float64)

length = min(len(a), len(b))

# Compute the energy stored in the sound array given
def compute_energy(length, array):
    energy = 0
    for i in range(length):
        energy = energy + array[i] * array[i]

    return energy



steps = 1
step_length = int(length / steps)

for i in range(steps):

    section_a = a[i*step_length : (i+1)*step_length]
    section_b = b[i*step_length : (i+1)*step_length]
    a_energy = compute_energy(step_length, section_a)
    b_energy = compute_energy(step_length, section_b)
    
    # Define variables from the algorithm
    
    R = 12 
    m = a_energy / b_energy
    k = R * (m + 1) / (m - 1)
    l = (1 / (m - 1)) * ((4*m*R*R)/(m-1))
    rootl = np.sqrt(l)
    
    print("Circle center: ", k)
    print("Circle radius: ", rootl)
    print()
