import numpy as np
import tensorflow as tf
import scipy.io.wavfile
import scipy.misc
import math
import os
import re
import h5py
import matplotlib.pyplot as plt

window_size = 2048
wavs = np.zeros(window_size)
test_wavs = np.zeros(window_size)

f= open(".\\sound_files\\non_speech\\files.txt","r")
non_speech_files = f.readlines()
i=0
# for file in non_speech_files:
#     print("working on " + file)
#     file = file.rstrip()
#     rate, wav_data = scipy.io.wavfile.read(".\\sound_files\\non_speech\\"+  file)
#     if rate != 44100:
#         print("input file does not have 44.1kHz sample rate: " + file)

#     min_length = len(wav_data)
#     wav_data = wav_data[:min_length]

#     padding = min_length % window_size
#     if padding == 0:
#         padding = window_size

#     wav_data = np.append(wav_data, np.zeros(padding))
#     if(i%30 != 0):
#         wavs = np.append(wavs, wav_data)
#     else:
#         test_wavs = np.append(test_wavs,wav_data)
#     i=i+1

# scipy.io.wavfile.write(".\\sound_files\\no_speech.wav",44100,wavs)
# scipy.io.wavfile.write(".\\sound_files\\no_speech_test.wav",44100,test_wavs)

rate, wav_data = scipy.io.wavfile.read(".\\sound_files\\room-44100.wav")
if rate != 44100:
    print("input file does not have 44.1kHz sample rate: " + file)
min_length = len(wav_data)
wav_data = wav_data[:min_length]

print("sizes1")
print(len(wav_data))
print(len(test_wavs))
print(len(wavs))
test_wavs = np.append(test_wavs,wav_data[0:8000*window_size])
wavs = np.append(wavs,wav_data[8000*window_size:len(wav_data)])
print("sizes")
print(len(wav_data))
print(len(test_wavs))
print(len(wavs))

scipy.io.wavfile.write(".\\sound_files\\speech.wav",44100,wavs)
scipy.io.wavfile.write(".\\sound_files\\speech_test.wav",44100,test_wavs)

