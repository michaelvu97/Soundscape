#!/Library/Frameworks/Python.framework/Versions/3.6/bin/python3
import numpy as np
import math
import scipy.io.wavfile
import pyaudio
from scipy import signal
import wave
import pygame
import struct
import time
import draw_arrow
from scipy.fftpack import fft
import utils
import sys
sys.path.append("../../ML")
from vad_test import VAD
from rVAD_custom import VoiceActivityDetector


#rate, a = scipy.io.wavfile.read("/Users/alexmertens/Desktop/School/School/Capstone/code/Soundscape/alex/Sounds/Trimmed/honk mic 3 trimmed.wav")
#rate, b = scipy.io.wavfile.read("/Users/alexmertens/Desktop/School/School/Capstone/code/Soundscape/alex/Sounds/Trimmed/honk mic 4 trimmed.wav")
#rate, c = scipy.io.wavfile.read("/Users/alexmertens/Desktop/School/School/Capstone/code/Soundscape/alex/Sounds/Trimmed/honk mic 1 trimmed.wav")
##rate, a = scipy.io.wavfile.read("./honk mic 3.wav")
##rate, b = scipy.io.wavfile.read("./honk mic 4.wav")
#
#a = a.astype(np.float64)
#b = b.astype(np.float64)
#c = c.astype(np.float64)
#
#length = min(len(a), len(b), len(c))

# Compute the energy stored in the sound array given
def compute_energy(data_channels):
    if data_channels.ndim == 2:
        return np.sum(np.square(data_channels), axis=1)
    else:
        return np.sum(np.square(data_channels))


form_1 = pyaudio.paInt16 # 16-bit resolution
samp_rate = 44100 # 44.1kHz sampling rate
chunk = 8192 # 2^12 samples for buffer
height = 600
width = 1000

vad = VoiceActivityDetector(samp_rate)

BLACK = (  0,   0,   0)
WHITE = (255, 255, 255)
BLUE =  (  0,   0, 255)
GREEN = (  0, 255,   0)
RED =   (255,   0,   0)
audio = pyaudio.PyAudio() # create pyaudio instantiation
device_indexes = utils.get_device_indices()

# Initialize butterworth filter
nyq = 0.5 * samp_rate
low = 100 / nyq
high = 800 / nyq
den, num = signal.butter(2, [low, high], btype='band')
amp = 1
size = [width, height]
screen = pygame.display.set_mode(size)

streams = [
    audio.open(format = form_1,rate = samp_rate, channels = 1, \
                    input_device_index = device_index, input = True, \
                    frames_per_buffer=chunk)
        for device_index in device_indexes
    ]

done = False
while not done:    

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done=True

    screen.fill(BLACK)

    data = [np.fromstring(stream.read(chunk, exception_on_overflow=False), dtype=np.int16).astype(np.float64) for stream in streams]

    # Filtering
    data_filtered = signal.filtfilt(den, num, data)

    # Normalization
    data_filtered_normalized = data_filtered / 256

    # energies = utils.compute_energy(data_filtered_normalized)
    # print("Energies: " + str(energies.astype(np.int32)))

    length = min(len(data[0]), len(data[1]), len(data[2]), len(data[3]))

    steps = 1
    step_length = int(length / steps)
    
    for i in range(steps):
        windowed_data = data_filtered_normalized[:, i * step_length:(i + 1) * step_length]
        window_energy = utils.compute_energy(windowed_data)

        # Define variables from the algorithm
        
        R1 = 7.5 
        m1 = window_energy[0] / window_energy[1]
        k1 = R1 * (m1 + 1) / (m1 - 1)
        l1 = (1 / (m1 - 1)) * ((4*m1*R1*R1)/(m1-1))
        rootl1 = np.sqrt(l1)
        
        R2 = 7.5
        m2 = window_energy[0] / window_energy[2]
        k2 = R2 * (m2 + 1) / (m2 - 1)
        l2 = (1 / (m2 - 1)) * ((4*m2*R2*R2)/(m2-1))
        rootl2 = np.sqrt(l2)
    
        centre2 = k2 - R2
        
        # Vector of predictions of whether each channel is voice
        is_voice = vad.is_speech(windowed_data)
        print(is_voice)
        draw_arrow.drawVoice(screen, is_voice[0], is_voice[2], is_voice[3], is_voice[1])
        pygame.display.update()
        # For now for testing, just examing channel 0 (A)
        # if is_voice[0]:
        #     print("VOICE")
        # else:
        #     print("NOT VOICE")
    
        #Solve for intersectin of circles source: http://paulbourke.net/geometry/circlesphere/
    
        #First get the distance between the circles and the radius of each circle
        d = math.sqrt((k1 - R2) ** 2 + (centre2 ** 2))
        r0 = rootl1
        r1 = rootl2
    
        if(d > (r0 + r1)):
            # print("NO SOLUTIONS")
            continue
        if(d < abs(r0 - r1)):
            # print("NO SOLUTIONS")
            continue
    
        A = (r0**2 - r1**2 + d**2) / (2 * d)
        h = math.sqrt(r0**2 - A**2)
    
        #Label each centre point
        x0 = k1
        y0 = 0
    
        x1 = R2
        y1 = centre2
    
        x2 = x0 + A * (x1 - x0) / d
        y2 = y0 + A * (y1 - y0) / d
    
        x3_1 = x2 + h * (y1 - y0) / d
        y3_1 = y2 - h * (x1 - x0) / d
        
        x3_2 = x2 - h * (y1 - y0) / d
        y3_2 = y2 + h * (x1 - x0) / d
    
        dist1 = math.sqrt(x3_1 ** 2 + y3_1 ** 2)
        dist2 = math.sqrt(x3_2 ** 2 + y3_2 ** 2)
    
        if(dist1 > dist2):
            angle = np.arctan(y3_1 / x3_1)
        else:
            angle = np.arctan(y3_2 / x3_2)
    
        angle = 180 * angle / math.pi
    
    #    print("Circle center: ", k1)
    #    print("Circle radius: ", rootl1)
    #    print("Circle equation: (x -",k1,")^2 + y^2 =",l1)
    #    print()
    #
    #    print("Circle center: ", R2, centre2)
    #    print("Circle radius: ", rootl2)
    #    print("Circle equation: (x -",R2,")^2 + (y -", centre2,")^2 =",l2)
    #    print()
    #
    #
    #    print("Distance between circles:", d)
    #    print()
    #
    #    print("Intersection 1:", x3_1, y3_1)
    #    print("Intersection 2:", x3_2, y3_2)
    #    print()
    
        # print("Angle of Sound:", angle)
        draw_arrow.drawArrow(screen, angle)
        pygame.display.update()
