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
noise_gate = 1e-3
height = 600
width = 1000



BLACK = (  0,   0,   0)
WHITE = (255, 255, 255)
BLUE =  (  0,   0, 255)
GREEN = (  0, 255,   0)
RED =   (255,   0,   0)


audio = pyaudio.PyAudio() # create pyaudio instantiation

size = [width, height]
screen = pygame.display.set_mode(size)

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

while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done=True

    data1 = stream1.read(chunk, exception_on_overflow=False)
    data2 = stream2.read(chunk, exception_on_overflow=False)

    data1_int = np.fromstring(data1, dtype=np.int16)
    data2_int = np.fromstring(data2, dtype=np.int16)

    chan1 = np.array(data1_int)
    chan2 = np.array(data2_int)

    chan1_fft_temp = fft(chan1)
    chan2_fft_temp = fft(chan2)

    chan1_fft = np.abs(chan1_fft_temp[0:chunk]) * 2 / (256 * chunk)
    chan2_fft = np.abs(chan2_fft_temp[0:chunk]) * 2 / (256 * chunk)

    chan1_energies = []
    chan2_energies = []

    for i in range(sources):
        chan1_energies.append(0)
        chan2_energies.append(0)

    for i in range(sources):
        start = int(i * (chunk / 2) / (sources))
        end = int((i + 1) * (chunk / 2) / (sources))

        for j in range(start, end):
            chan1_energies[i] = chan1_energies[i] + chan1_fft[j]*chan1_fft[j]
            chan2_energies[i] = chan2_energies[i] + chan2_fft[j]*chan2_fft[j]

        chan1_energies[i] = chan1_energies[i] / chunk
        chan2_energies[i] = chan2_energies[i] / chunk

    directions = [0] * sources
    screen.fill(WHITE)

    for i in range(sources):
        directions[i] = (chan1_energies[i] - chan2_energies[i]) / (chan1_energies[i] + chan2_energies[i])

        if (chan1_energies[i] + chan2_energies[i]) < noise_gate:
            directions[i] = 0

        horiz = directions[i] * width/2 + width / 2 - 25

        R = int(255 * i / sources)
        G = (R + 30) % 256
        B = (G + 85) % 256
        pygame.draw.rect(screen, (R, G, B), [horiz, int((height/sources) * i), 50, int(height/sources)])
    pygame.display.flip()

pygame.quit()
# loop through stream and append audio chunks to frame array
#for ii in range(0,int((samp_rate/chunk)*record_secs)):
#    data1 = stream1.read(chunk, exception_on_overflow=False)
#    data2 = stream2.read(chunk, exception_on_overflow=False)
#
#    data1_int = np.fromstring(data1, dtype=np.int16)
#    data2_int = np.fromstring(data2, dtype=np.int16)
#
#    combined = np.asarray(convert_to_2_channels(data1_int, data2_int), dtype=np.int16)
#
#    frames1.append(combined.tostring())
#    frames2.append(data1)
#
#print("finished recording")

# stop the stream, close it, and terminate the pyaudio instantiation
stream1.stop_stream()
stream1.close()
stream2.stop_stream()
stream2.close()
audio.terminate()

