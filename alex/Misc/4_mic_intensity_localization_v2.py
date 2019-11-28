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
# import sys
# sys.path.append("../../ML")
# from vad_test import VAD
from rVAD_custom import VoiceActivityDetector
import live_calibration

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

def get_left(channels):
    return channels[0]
def get_right(channels):
    return channels[1]
def get_forward(channels):
    return channels[2]
def get_back(channels):
    return channels[3]

def localize2(energy):
    """
    Energy order: [A, B, C, D] = [left, right, forward, backward]
    """
    max_energy = np.max(energy)
    relative_db = np.log(energy / max_energy) # Will crash with zeros

    threshold_db = -0.5

    is_loud = relative_db > threshold_db

    left_loud = get_left(is_loud)
    right_loud = get_right(is_loud)
    forward_loud = get_forward(is_loud)
    backward_loud = get_back(is_loud)

    # Unhandled case: opposite mics on the same axis are loud (probably because of 2 sound sources)
    if (left_loud and right_loud) or (forward_loud and backward_loud):
        return None

    # Lookup logic
    if forward_loud:
        if left_loud:
            return 90 + 45
        elif right_loud:
            return 90 - 45
        else:
            return 90
    elif backward_loud:
        if left_loud:
            return 270 - 45
        elif right_loud:
            return 270 + 45
        else:
            return 270
    else:
        if left_loud:
            return 180
        else:
            return 0

def localize(channel_left_energy, channel_forward_energy, channel_right_energy, channel_back_energy):
    left_db = 20.0 * np.log1p(channel_left_energy)
    forward_db = 20.0 * np.log1p(channel_forward_energy)
    right_db = 20.0 * np.log1p(channel_right_energy)
    back_db = 20.0 * np.log1p(channel_back_energy)

    threshold_db = 13.0

    # NOTE: This is ultra stupid

    diff_front_back = forward_db - back_db
    diff_left_right = left_db - right_db

    is_forward = diff_front_back > threshold_db
    is_back = -diff_front_back > threshold_db
    is_left = diff_left_right > threshold_db
    is_right = -diff_left_right > threshold_db

    if is_forward:
        base_angle = 90
        if is_right:
            base_angle -= 45
        elif is_left:
            base_angle += 45
        return base_angle
    elif is_back:
        base_angle = 270
        if is_right:
            base_angle += 45
        elif is_left:
            base_angle -= 45
        return base_angle
    elif is_right:
        base_angle = 0
        if is_forward:
            base_angle += 45
        elif is_back:
            base_angle += 360 - 45
        return base_angle
    elif is_left:
        base_angle = 180
        if is_forward:
            base_angle -= 45
        elif is_back:
            base_angle += 45
        return base_angle
    else:
        return None

form_1 = pyaudio.paInt16 # 16-bit resolution
samp_rate = 44100 # 44.1kHz sampling rate
chunk = 4196 #8192 # 2^12 samples for buffer
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

calibration = live_calibration.LiveCalibration(len(device_indexes))

# Initialize butterworth filter
nyq = 0.5 * samp_rate
low = 300 / nyq
high = 8000 / nyq
den, num = signal.butter(2, [low, high], btype='bandpass')

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

    data = np.stack([np.fromstring(stream.read(chunk, exception_on_overflow=False), dtype=np.int16).astype(np.float64) for stream in streams])
    
    # Calibrate
    calibration_mode = False
    if (calibration_mode):
        gain_correction = calibration.get_gain_correction(utils.compute_energy(data))
        print(gain_correction)
        data_gain_corrected = data * gain_correction
    else:
        # Hardcoded calibration
        data_gain_corrected = data * [[1.1041753],[1.2040477],[1.1236639],[1.       ]]

    # Filtering
    data_filtered = signal.filtfilt(den, num, data_gain_corrected)
    data_unfiltered = data_gain_corrected

    # Normalization
    data_filtered_normalized = data_filtered / 256

    # energies = utils.compute_energy(data_filtered_normalized)
    # print("Energies: " + str(energies.astype(np.int32)))

    length = min([len(x) for x in data])

    steps = 1
    step_length = int(length / steps)
    
    for i in range(steps):
        windowed_data_unfiltered = data_unfiltered[:, i * step_length:(i + 1) * step_length]
        windowed_data = data_filtered_normalized[:, i * step_length:(i + 1) * step_length]
        window_energy = utils.compute_energy(windowed_data)

        # Define variables from the algorithm
        """
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
        """
    
        #Solve for intersectin of circles source: http://paulbourke.net/geometry/circlesphere/
    
        #First get the distance between the circles and the radius of each circle
        """
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
        """
        draw_arrow.drawMicLevels(screen, window_energy)
    
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

        # Localization using energy difference

        # Vector of predictions of whether each channel is voice
        voice_confidence = vad.is_speech(windowed_data_unfiltered)

        # print(is_voice)
        draw_arrow.drawVoice(screen, voice_confidence[0], voice_confidence[2], voice_confidence[1], voice_confidence[3])

        angle = localize2(window_energy)

        draw_arrow.drawArrow(screen, angle)
        pygame.display.update()
