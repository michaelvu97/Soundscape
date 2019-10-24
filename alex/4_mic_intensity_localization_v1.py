#!/Library/Frameworks/Python.framework/Versions/3.6/bin/python3
import pyaudio
from scipy import signal
import wave
import numpy as np
import pygame
import struct
import time
import math
from draw_arrow import drawArrow, drawArrowNone, drawMicLevels
from scipy.fftpack import fft
from get_device_indices import get_device_indices

def set_device_indices():
    indices = get_device_indices();
    
    #microphones are in pairs like this
    dev1_index = indices[0];
    dev4_index = indices[1];
    
    dev2_index = indices[2]
    dev3_index = indices[3]

def get_colour_of_frequency(frequency):
    return (0, 0, 0)

def convert_to_2_channels(left, right):
    stereo = []
    for i in range(len(left)):
        stereo.append(left[i])
        stereo.append(right[i])
    return(stereo)

def xy_to_angle_rads(x, y):
    theta = 0
    if (x > 0):
        theta = math.atan(y/x)
        if theta < 0:
            theta += 2 * math.pi
    elif (x < 0):
        theta = math.atan(y/x) + math.pi
    else:
        if (y > 0):
            theta = math.pi / 2
        elif (y < 0):
            theta = math.pi / -2
        else:
            theta = 0
    return theta


chan1_reduction = 8
round_factor = 1
form_1 = pyaudio.paInt16 # 16-bit resolution
chans = 1 # 1 channel
samp_rate = 44100 # 44.1kHz sampling rate
chunk = 16384 # 2^12 samples for buffer
record_secs = 10 # seconds to record
dev1_index = 2 # device index found by p.get_device_info_by_index(ii)
dev2_index = 4 # device index found by p.get_device_info_by_index(ii)
dev3_index = 5 # device index found by p.get_device_info_by_index(ii)
dev4_index = 3 # device index found by p.get_device_info_by_index(ii)
wav_output_filename1 = 'test1.wav' # name of .wav file
wav_output_filename2 = 'test2.wav' # name of .wav file
frequencies = []
sources = 50
max_energy = 0
min_energy = 1
noise_gate = 1e-4
height = 600
width = 1000



BLACK = (  0,   0,   0)
WHITE = (255, 255, 255)
BLUE =  (  0,   0, 255)
GREEN = (  0, 255,   0)
RED =   (255,   0,   0)


set_device_indices()
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

stream3 = audio.open(format = form_1,rate = samp_rate,channels = 1, \
                    input_device_index = dev3_index,input = True, \
                    frames_per_buffer=chunk)

stream4 = audio.open(format = form_1,rate = samp_rate,channels = 1, \
                    input_device_index = dev4_index,input = True, \
                    frames_per_buffer=chunk)

print("recording")
frames1 = []
frames2 = []

done = False
horiz = 0
vert = 0



# Initialize butterworth filter
nyq = 0.5 * samp_rate
low = 100/nyq
high = 500/nyq
b, a = signal.butter(1, [low, high], btype='band')
amp =1


while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done=True
    screen.fill(BLACK)

    start = time.time() * 1000

    data1 = stream1.read(chunk, exception_on_overflow=False)
    data2 = stream2.read(chunk, exception_on_overflow=False)
    data3 = stream3.read(chunk, exception_on_overflow=False)
    data4 = stream4.read(chunk, exception_on_overflow=False)

    end = time.time() * 1000
    taken = end - start
    print(taken)

    data1_int = np.fromstring(data1, dtype=np.int16)
    data2_int = np.fromstring(data2, dtype=np.int16)
    data3_int = np.fromstring(data3, dtype=np.int16)
    data4_int = np.fromstring(data4, dtype=np.int16)


    data4_int = signal.filtfilt(b, a, data4_int)
    data3_int = signal.filtfilt(b, a, data3_int)
    data2_int = signal.filtfilt(b, a, data2_int)
    data1_int = signal.filtfilt(b, a, data1_int)

    chan1 = np.array(data1_int)
    chan2 = np.array(data2_int)
    chan3 = np.array(data3_int)
    chan4 = np.array(data4_int)

    chan1_fft_temp = fft(chan1)
    chan2_fft_temp = fft(chan2)
    chan3_fft_temp = fft(chan3)
    chan4_fft_temp = fft(chan4)

    chan1_fft = np.abs(chan1_fft_temp[0:chunk]) * 2 / (256 * chunk)
    chan2_fft = np.abs(chan2_fft_temp[0:chunk]) * 2 / (256 * chunk)
    chan3_fft = np.abs(chan3_fft_temp[0:chunk]) * 2 / (256 * chunk)
    chan4_fft = np.abs(chan4_fft_temp[0:chunk]) * 2 / (256 * chunk)

    chan1_energies = []
    chan2_energies = []
    chan3_energies = []
    chan4_energies = []

    for i in range(sources):
        chan1_energies.append(0)
        chan2_energies.append(0)
        chan3_energies.append(0)
        chan4_energies.append(0)

    for i in range(sources):
        start = int(i * (chunk / 2) / (sources))
        end = int((i + 1) * (chunk / 2) / (sources))

        for j in range(start, end):
            chan1_energies[i] = chan1_energies[i] + chan1_fft[j]*chan1_fft[j]
            chan2_energies[i] = chan2_energies[i] + chan2_fft[j]*chan2_fft[j]
            chan3_energies[i] = chan3_energies[i] + chan3_fft[j]*chan3_fft[j]
            chan4_energies[i] = chan4_energies[i] + chan4_fft[j]*chan4_fft[j]

        chan1_energies[i] = chan1_energies[i] / (chan1_reduction * chunk)
        chan2_energies[i] = chan2_energies[i] / chunk
        chan3_energies[i] = chan3_energies[i] / chunk
        chan4_energies[i] = chan4_energies[i] / chunk

    directions = [0] * sources
    directions2 = [0] * sources
    #screen.fill(WHITE)

        

    for i in range(int(sources/4)):
        directions[i] = (chan1_energies[i] - chan2_energies[i]) / (chan1_energies[i] + chan2_energies[i])
        directions2[i] = (chan3_energies[i] - chan4_energies[i]) / (chan3_energies[i] + chan4_energies[i])

        if (chan1_energies[i] + chan2_energies[i]) < noise_gate:
            directions[i] = 0

        if (chan3_energies[i] + chan4_energies[i]) < noise_gate:
            directions2[i] = 0

        horiz = directions[i] * width/2 + width / 2 - 25
        vert = directions2[i] * height/2 + height / 2 - 25

        horiz = int(round_factor * round(horiz/round_factor))
        vert = int(round_factor * round(vert/round_factor))
        
        frequency = frequencies[int(2048 * (i / sources))]
        colour = get_colour_of_frequency(frequency)
        #R = int(255 * i / sources)
        #G = (R + 30) % 256
        #B = (G + 85) % 20
        #pygame.draw.rect(screen, colour, [horiz, vert, 10, 10])
    # for now: we will choose the largest frequency amplitude as the angle
    directions_energy = []
    for i in range (int(sources/4)):
        #directions_energy.append(directions[i] ** 2 + directions2[i] ** 2)
        directions_energy.append(chan1_energies[i] + chan2_energies[i] + chan3_energies[i] + chan4_energies[i])
    
    max_energy_threshold = 0.01 # TODO

    max_energy = max(directions_energy) 
    max_energy_index = directions_energy.index(max(directions_energy))
    drawMicLevels(screen, [chan1_energies[max_energy_index], chan2_energies[max_energy_index], chan3_energies[max_energy_index], chan4_energies[max_energy_index]])   
    if (max_energy > max_energy_threshold):
        # Determine angle (this is a small TODO)
        angle = xy_to_angle_rads(directions[max_energy_index], directions2[max_energy_index]) * 180 / math.pi
        print("Angle is", angle)
        drawArrow(screen, angle)
        print("Direction 1: ", directions[max_energy_index], " Direction 2: ", directions2[max_energy_index])
        #print("levels @bin: " + str(max_energy_index) + " Channel 1: " + str(chan1_energies[max_energy_index]) + " Channel 2:" + str(chan2_energies[max_energy_index]) + " Channel 3:" + str(chan3_energies[max_energy_index]) + " Channel4: " + str(chan4_energies[max_energy_index]) + "\n");
    else:
        drawArrowNone(screen)    
    pygame.display.update()
#pygame.display.flip()

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
