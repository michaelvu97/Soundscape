
import numpy as np
import scipy.io
import scipy.io.wavfile as wav
import scipy.signal

target_rate = 44100


# Resample audio
source_rate = 16000
# source_rate, source_audio = wav.read("./room.wav")
print("Audio read")

# target_num_samples = int(target_rate * len(source_audio) / float(source_rate))
# print("target samples: " + str(target_num_samples))
# res_audio = scipy.signal.resample_poly(source_audio, target_rate/100, source_rate/100)
print("Audio resampled")
# wav.write("./room-44100.wav", target_rate, res_audio)
res_audio = None

print("Audio written")

arr = scipy.io.loadmat('room.mat')['y_label']

# if (len(arr) != len(source_audio)):
#     print("Lengths mismatch! ," + str(len(arr)) + ", " + str(len(source_audio)))

source_audio = None

target_length = int(len(arr) * float(target_rate) / float(source_rate))

res_label = [0.0] * target_length

target_sample_to_source_sample = float(source_rate) / float(target_rate)

for i in range(target_length):
    source_idx = int(i * target_sample_to_source_sample)
    res_label[i] = arr[source_idx]

np.save('room.npy', res_label)
