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
from draw_arrow import drawArrow, drawArrowNone, drawMicLevels
from scipy.fftpack import fft
from utils import get_device_indices


def set_device_indices():
    indices = get_device_indices();

    print("IM HERE")
    
    #microphones are in pairs like this
    dev1_index = indices[0];
    print(indices[0], dev1_index)
    dev4_index = indices[1];
    
    dev2_index = indices[2]
    dev3_index = indices[3]


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
def compute_energy(length, array):
    energy = 0
    for i in range(length):
        energy = energy + array[i] * array[i]

    return energy



form_1 = pyaudio.paInt16 # 16-bit resolution
samp_rate = 44100 # 44.1kHz sampling rate
chunk = 8192 # 2^12 samples for buffer
dev1_index = 2 # device index found by p.get_device_info_by_index(ii)
dev2_index = 4 # device index found by p.get_device_info_by_index(ii)
dev3_index = 5 # device index found by p.get_device_info_by_index(ii)
dev4_index = 3 # device index found by p.get_device_info_by_index(ii)
height = 600
width = 1000


BLACK = (  0,   0,   0)
WHITE = (255, 255, 255)
BLUE =  (  0,   0, 255)
GREEN = (  0, 255,   0)
RED =   (255,   0,   0)
audio = pyaudio.PyAudio() # create pyaudio instantiation
set_device_indices()
print(dev1_index, dev2_index, dev3_index)

# Initialize butterworth filter
nyq = 0.5 * samp_rate
low = 100/nyq
high = 800/nyq
den, num = signal.butter(2, [low, high], btype='band')
amp =1
size = [width, height]
screen = pygame.display.set_mode(size)
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

while(True):    

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done=True
    screen.fill(BLACK)
    data1 = stream1.read(chunk, exception_on_overflow=False)
    data2 = stream2.read(chunk, exception_on_overflow=False)
    data3 = stream3.read(chunk, exception_on_overflow=False)
    data4 = stream4.read(chunk, exception_on_overflow=False)

    d = np.fromstring(data1, dtype=np.int16)
    a = np.fromstring(data2, dtype=np.int16)
    c = np.fromstring(data3, dtype=np.int16)
    b = np.fromstring(data4, dtype=np.int16)

    a = signal.filtfilt(den, num, a)
    b = signal.filtfilt(den, num, b)
    c = signal.filtfilt(den, num, c)
    d = signal.filtfilt(den, num, d)

    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    c = np.array(c, dtype=np.float64)
    d = np.array(d, dtype=np.float64)

    a = a / 256
    b = b / 256
    c = c / 256
    d = d / 256

    aenergy = compute_energy(len(a), a)
    benergy = compute_energy(len(b), b)
    cenergy = compute_energy(len(c), c)
    denergy = compute_energy(len(d), d)

#    print("A:", aenergy)
#    print("B:", benergy)
#    print("C:", cenergy)
#    print("D:", denergy)
#    print()

    length = min(len(a), len(b), len(c))


    steps = 1
    step_length = int(length / steps)
    
    for i in range(steps):
    #    print(i)
    #    print(len(a))
    #    print((i+1)*step_length)
        section_a = a[i*step_length : (i+1)*step_length]
        section_b = b[i*step_length : (i+1)*step_length]
        section_c = c[i*step_length : (i+1)*step_length]
        a_energy = compute_energy(step_length, section_a)
        b_energy = compute_energy(step_length, section_b)
        c_energy = compute_energy(step_length, section_c)
        
        # Define variables from the algorithm
        
        R1 = 7.5 
        m1 = a_energy / b_energy
        k1 = R1 * (m1 + 1) / (m1 - 1)
        l1 = (1 / (m1 - 1)) * ((4*m1*R1*R1)/(m1-1))
        rootl1 = np.sqrt(l1)
        
        R2 = 7.5
        m2 = a_energy / c_energy
        k2 = R2 * (m2 + 1) / (m2 - 1)
        l2 = (1 / (m2 - 1)) * ((4*m2*R2*R2)/(m2-1))
        rootl2 = np.sqrt(l2)
    
        centre2 = k2 - R2
    
        
    
        #Solve for intersectin of circles source: http://paulbourke.net/geometry/circlesphere/
    
        #First get the distance between the circles and the radius of each circle
        d = math.sqrt((k1 - R2) ** 2 + (centre2 ** 2))
        r0 = rootl1
        r1 = rootl2
    
        if(d > (r0 + r1)):
            print("NO SOLUTIONS")
            continue
        if(d < abs(r0 - r1)):
            print("NO SOLUTIONS")
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
    
        print("Angle of Sound:", angle)
        drawArrow(screen, angle)
        pygame.display.update()
