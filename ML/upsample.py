
import numpy as np
import scipy.io
import scipy.io.wavfile as wav
import scipy.signal

target_rate = 44100


# Resample audio
source_rate, source_audio = wav.read("./room.wav")
res_audio = scipy.signal.resample(source_audio, target_rate)
wav.write("./room-44100.wav", target_rate, res_audio)
res_audio = None

print("Audio written")

arr = scipy.io.loadmat('room.mat')['y_label']

if (len(arr) != len(source_audio)):
    print("Lengths mismatch! ," + str(len(arr)) + ", " + str(len(source_audio)))

source_audio = None

target_length = int(len(arr) * float(target_rate) / float(source_rate))

res_label = [0.0] * target_length

target_sample_to_source_sample = int(float(source_rate) / float(target_length))

for i in range(target_length):
    source_idx = i * target_sample_to_source_sample
    res_label[i] = arr[source_idx]

print(len(res))

np.save('room.npy', res_label)
