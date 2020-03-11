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

f= open(".\\sound_files\\non_speech\\files.txt","r")
non_speech_files = f.readlines()
for file in non_speech_files:
    print("working on " + file)
    file = file.rstrip()
    rate, wav_data = scipy.io.wavfile.read(".\\sound_files\\non_speech\\"+  file)
    if rate != 44100:
        print("input file does not have 44.1kHz sample rate: " + file)

    min_length = len(wav_data)
    wav_data = wav_data[:min_length]

    padding = min_length % window_size
    if padding == 0:
        padding = window_size

    wav_data = np.append(wav_data, np.zeros(padding))

    wavs = np.append(wavs, wav_data)

scipy.io.wavfile.write(".\\sound_files\\no_speech.wav",44100,wavs)
