#!/Library/Frameworks/Python.framework/Versions/3.6/bin/python3
import numpy as np
import math
import scipy.io.wavfile



rate, a = scipy.io.wavfile.read("./Honk Trimmed mic 3.wav")
rate, b = scipy.io.wavfile.read("./Honk Trimmed mic 4.wav")

a = a.astype(np.float64)
b = b.astype(np.float64)

print(len(a))
