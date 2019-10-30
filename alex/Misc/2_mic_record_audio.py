#!/Library/Frameworks/Python.framework/Versions/3.6/bin/python3
import pyaudio
import wave
import numpy as np
import pygame
import struct
import time
from scipy.fftpack import fft


def convert_to_2_channels(left, right):
    stereo = []
    for i in range(len(left)):
        stereo.append(left[i])
        stereo.append(right[i])
    return(stereo)


form_1 = pyaudio.paInt16 # 16-bit resolution
chans = 1 # 1 channel
samp_rate = 44100 # 44.1kHz sampling rate
chunk = 4096 # 2^12 samples for buffer
record_secs = 10 # seconds to record
dev1_index = 4 # device index found by p.get_device_info_by_index(ii)
dev2_index = 5 # device index found by p.get_device_info_by_index(ii)
wav_output_filename1 = 'test1.wav' # name of .wav file
wav_output_filename2 = 'test2.wav' # name of .wav file
frequencies = []
sources = 100
max_energy = 0
min_energy = 1
noise_gate = 1e-10


BLACK = (  0,   0,   0)
WHITE = (255, 255, 255)
BLUE =  (  0,   0, 255)
GREEN = (  0, 255,   0)
RED =   (255,   0,   0)


audio = pyaudio.PyAudio() # create pyaudio instantiation


# Pre fill an array with the frequency data  corresponding to each index of the fft
for i in range(int(chunk/2)):
	temp = int(samp_rate * i / chunk)
	frequencies.append(temp)


# create pyaudio stream
stream1 = audio.open(format = form_1,rate = samp_rate,channels = 1, \
                    input_device_index = dev1_index,input = True, \
                    frames_per_buffer=chunk)

stream2 = audio.open(format = form_1,rate = samp_rate,channels = 1, \
                    input_device_index = dev2_index,input = True, \
                    frames_per_buffer=chunk)
print("recording")
frames1 = []
frames2 = []

done = False
horiz = 0


# loop through stream and append audio chunks to frame array
for ii in range(0,int((samp_rate/chunk)*record_secs)):
    data1 = stream1.read(chunk, exception_on_overflow=False)
    data2 = stream2.read(chunk, exception_on_overflow=False)

    data1_int = np.fromstring(data1, dtype=np.int16)
    data2_int = np.fromstring(data2, dtype=np.int16)

    combined = np.asarray(convert_to_2_channels(data1_int, data2_int), dtype=np.int16)

    frames1.append(combined.tostring())
    frames2.append(data1)

print("finished recording")
print(len(frames1))
print(len(frames2))
# stop the stream, close it, and terminate the pyaudio instantiation
stream1.stop_stream()
stream1.close()
stream2.stop_stream()
stream2.close()
audio.terminate()

# save the audio frames as .wav file
wavefile1 = wave.open(wav_output_filename1,'wb')
wavefile1.setnchannels(2)
wavefile1.setsampwidth(audio.get_sample_size(form_1))
wavefile1.setframerate(samp_rate)
wavefile1.writeframes(b''.join(frames1))
wavefile1.close()


wavefile2 = wave.open(wav_output_filename2,'wb')
wavefile2.setnchannels(1)
wavefile2.setsampwidth(audio.get_sample_size(form_1))
wavefile2.setframerate(samp_rate)
wavefile2.writeframes(b''.join(frames2))
wavefile2.close()
