#!/Library/Frameworks/Python.framework/Versions/3.6/bin/python3
import pyaudio
import wave
import numpy as np
import pygame
import struct
import time
from scipy.fftpack import fft
from utils import get_device_indices

form_1 = pyaudio.paInt16 # 16-bit resolution
chans = 1 # 1 channel
samp_rate = 44100 # 44.1kHz sampling rate
chunk = 22050 # 2^12 samples for buffer
record_secs = 5 # seconds to record
dev1_index = 0 # device index found by p.get_device_info_by_index(ii)
dev2_index = 0 # device index found by p.get_device_info_by_index(ii)
dev3_index = 0 # device index found by p.get_device_info_by_index(ii)
dev4_index = 0 # device index found by p.get_device_info_by_index(ii)
wav_output_filename = 'test.wav' # name of .wav file
wav_output_filename1 = 'test1.wav' # name of .wav file
wav_output_filename2 = 'test2.wav' # name of .wav file
wav_output_filename3 = 'test3.wav' # name of .wav file
wav_output_filename4 = 'test4.wav' # name of .wav file
frequencies = []
sources = 100
max_energy = 0
min_energy = 1
noise_gate = 1e-10


def set_device_indices():
    indices = get_device_indices();

    print("IM HERE")
    
    #microphones are in pairs like this
    dev1_index = indices[0];
    print(indices[0], dev1_index)
    dev4_index = indices[1];
    
    dev2_index = indices[2]
    dev3_index = indices[3]

audio = pyaudio.PyAudio()

set_device_indices()
# create pyaudio stream
input("Press enter when sound is near source 1...")
average1 = 0
stream1 = audio.open(format = form_1,rate = samp_rate,channels = 1, \
                    input_device_index = dev1_index,input = True, \
                    frames_per_buffer=chunk)
data1 = stream1.read(chunk, exception_on_overflow=False)
data1_int = np.fromstring(data1, dtype=np.int16)
chan1 = np.array(data1_int)
chan1 = chan1 / 256
for i in chan1:
    average1 = average1 + i*i
average1 = average1/chunk;

input("Press enter when sound is near source 2...")
average2 = 0
stream2 = audio.open(format = form_1,rate = samp_rate,channels = 1, \
                    input_device_index = dev2_index,input = True, \
                    frames_per_buffer=chunk)
data2 = stream2.read(chunk, exception_on_overflow=False)
data2_int = np.fromstring(data2, dtype=np.int16)
chan2 = np.array(data2_int)
chan2 = chan2 / 256
for i in chan2:
    average2 = average2 + i*i
average2 = average2/chunk;

input("Press enter when sound is near source 3...")
average3 = 0
stream3 = audio.open(format = form_1,rate = samp_rate,channels = 1, \
                    input_device_index = dev3_index,input = True, \
                    frames_per_buffer=chunk)
data3 = stream3.read(chunk, exception_on_overflow=False)
data3_int = np.fromstring(data2, dtype=np.int16)
chan3 = np.array(data3_int)
chan3 = chan3 / 256
for i in chan3:
    average3 = average3 + i*i
average3 = average3/chunk;

input("Press enter when sound is near source 4...")
average4 = 0
stream4 = audio.open(format = form_1,rate = samp_rate,channels = 1, \
                    input_device_index = dev4_index,input = True, \
                    frames_per_buffer=chunk)
data4 = stream4.read(chunk, exception_on_overflow=False) 
data4_int = np.fromstring(data2, dtype=np.int16)
chan4 = np.array(data4_int)
chan4 = chan4 / 256
for i in chan4:
    average4 = average4 + i*i
average4 = average4/chunk;

minimum_av=numpy.minimum(average1,average2,average3,average4)
average1 = average1/minimum
average2 = average2/minimum
average3 = average3/minimum
average4 = average4/minimum


f= open("mic_values.txt","w+")
f.write('{0}\n'.format(average1))
f.write('{0}\n'.format(average2))
f.write('{0}\n'.format(average3))
f.write('{0}\n'.format(average4))
