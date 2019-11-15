import pyaudio
import pygame
import sys
import numpy as np
sys.path.append("../../ML")
from vad_test import VAD

vad = VAD("../../ML/saved_models/model.h5")

audio = pyaudio.PyAudio()

frames_to_get = 4410

stream1 = audio.open(format = pyaudio.paInt16, rate = 44100, channels = 1, \
                    input_device_index = 1, input = True, \
                    frames_per_buffer=frames_to_get)

gain = 2.0

while(True):
    data = stream1.read(frames_to_get, exception_on_overflow=False)

    d = np.fromstring(data, dtype=np.int16) * gain

    if vad.is_voice(d):
        print("VOICE")
    else:
        print("NOT VOICE")