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
import live_calibration
from angle_smoother import AngleSmoother


def get_left(channels):
    return channels[0]
def get_right(channels):
    return channels[1]
def get_forward(channels):
    return channels[2]
def get_back(channels):
    return channels[3]

def localize(energy):
    """
    Energy order: clockwise starting at 3 o clock (0 degrees)
    """
    max_energy = np.max(energy)
    if (max_energy <= 0.00001):
        max_energy = 0.000001
    relative_db = np.log(energy / max_energy) # Will crash with zeros

    # print(relative_db)
    threshold_db = -0.5

    is_loud = relative_db > threshold_db

    if (np.sum(is_loud) > 2):
        return None # Experimental

#    left_loud = get_left(is_loud)
#    right_loud = get_right(is_loud)
#    forward_loud = get_forward(is_loud)
#    backward_loud = get_back(is_loud)

    # temp
    num_devices = 5

    loud = [0,0,0,0,0]
    for n in range(num_devices):
        loud[n] = is_loud[n]

    # Unhandled case: opposite mics on the same axis are loud (probably because of 2 sound sources)
    #if (left_loud and right_loud) or (forward_loud and backward_loud):
    #    return None

    # Lookup logic
    #if forward_loud:
    #    if left_loud:
    #        return 90 + 45
    #    elif right_loud:
    #        return 90 - 45
    #    else:
    #        return 90
    #elif backward_loud:
    #    if left_loud:
    #        return 270 - 45
    #    elif right_loud:
    #        return 270 + 45
    #    else:
    #        return 270
    #else:
    #    if left_loud:
    #        return 180
    #    else:
    #        return 0

    for n in range(num_devices):
        angle = n * (360 / (num_devices))
        incr = 360 / (2*num_devices)
        if(loud[n]):

            next_ind = (n + 1) % num_devices
            if(loud[next_ind]):
                return (angle + incr) % 360

            prev_ind = (n - 1) % num_devices
            if(loud[prev_ind]):
                return (angle - incr) % 360


            print(n, prev_ind, next_ind)
            return angle % 360

    return None


form_1 = pyaudio.paInt16 # 16-bit resolution
samp_rate = 44100 # 44.1kHz sampling rate
chunk = 2048 #8192 # 2^12 samples for buffer MUST BE 2048 FOR VAD
height = 600
width = 1000
vad = VAD("../../ML/saved_models/model_15_features_test.h5")
smoother = AngleSmoother();

BLACK = (  0,   0,   0)
WHITE = (255, 255, 255)
BLUE =  (  0,   0, 255)
GREEN = (  0, 255,   0)
RED =   (255,   0,   0)
audio = pyaudio.PyAudio() # create pyaudio instantiation
device_indexes = utils.get_device_indices()
num_devices = len(device_indexes)

calibration = live_calibration.LiveCalibration(len(device_indexes))

# Initialize butterworth filter
nyq = 0.5 * samp_rate
# low = 300 / nyq
# high = 8000 / nyq
low = 2700 / nyq
high = 6000 / nyq
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

print (len(streams))

steps = 1
step_length = int(chunk / steps)

windowed_data_unfiltered_delay_1_frame = np.zeros((num_devices, step_length))
windowed_data_unfiltered_delay_2_frame = np.zeros((num_devices, step_length))

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
        # Hardcoded calibration for 2 mics atm
        data_gain_corrected = data * [[1.182931 ],[1.422893 ],[1.2982392],[1.1       ],[1.4089305]]
        # data_gain_corrected = data * [[1.],[1.8]]  
        # data_gain_corrected = data

    # Filtering
    data_filtered = signal.filtfilt(den, num, data_gain_corrected)
    data_unfiltered = data_gain_corrected

    # Normalization
    data_filtered_normalized = data_filtered / 256

    # energies = utils.compute_energy(data_filtered_normalized)
    # print("Energies: " + str(energies.astype(np.int32)))

    length = min([len(x) for x in data])

    
    
    for i in range(steps):
        windowed_data_unfiltered = data_unfiltered[:, i * step_length:(i + 1) * step_length]
        windowed_data = data_filtered_normalized[:, i * step_length:(i + 1) * step_length]
        window_energy = utils.compute_energy(windowed_data)

        draw_arrow.drawMicLevels(screen, window_energy)
    

        # Localization using energy difference

        # Vector of predictions of whether each channel is voice
        voice_confidence = vad.is_voice_vectorized(windowed_data_unfiltered,
            windowed_data_unfiltered_delay_1_frame,
            windowed_data_unfiltered_delay_2_frame)

        voice_triggered = np.any(voice_confidence)

        # print(is_voice)
        #draw_arrow.drawVoice(screen, voice_confidence[0], voice_confidence[2], voice_confidence[1], voice_confidence[3])
        #draw_arrow.drawVoice(screen, voice_confidence[0], voice_confidence[2])

        angle = localize(window_energy)
        smoother.update(angle)
        angle = smoother.getSmoothedAngle()

        if not voice_triggered:
            angle = None

        # OFfset the angle to align the 1st mic with forward
        if angle is not None:
            angle = (angle + 0) % 360

        draw_arrow.drawArrow(screen, angle)
        pygame.display.update()

        windowed_data_unfiltered_delay_2_frame = windowed_data_unfiltered_delay_1_frame
        windowed_data_unfiltered_delay_1_frame = windowed_data_unfiltered
