import numpy as np
import tensorflow as tf
import scipy.io.wavfile
import scipy.misc
import math
import os
import re
import h5py
import matplotlib.pyplot as plt
from vad_test import VAD

NUM_HISTORY = 3
NUM_FEATURES = 5

def MergeData(window_size):
    file_list = [
        "./sound_files/speech_test",
        "./sound_files/no_speech_test"
    ]

    wavs = np.zeros(window_size)
    labels = np.zeros(window_size)
    for file in file_list:
        rate, wav_data = scipy.io.wavfile.read(file + ".wav")
        print(file)
        if rate != 44100:
            print("input file does not have 44.1kHz sample rate: " + file)
            return None

        if(file != "./sound_files/no_speech_test"):
            y_labels = np.load(file + ".npy").astype(np.float32)
            y_labels = y_labels[0:8000*window_size]
            y_labels = np.append(np.zeros(window_size),y_labels)
            print("LENGTH")
            print(len(wav_data))
            print(len(y_labels))
            min_length = min(len(y_labels), len(wav_data))
            y_labels = y_labels[:min_length]
            wav_data = wav_data[:min_length].astype(np.float32)

            padding = min_length % window_size
            if padding == 0:
                padding = window_size

            wav_data = np.append(wav_data, np.zeros(padding))
            y_labels = np.append(y_labels, np.zeros(padding))
        else:
            min_length = len(wav_data)
            wav_data = wav_data[:min_length].astype(np.float32)

            padding = min_length % window_size
            if padding == 0:
                padding = window_size

            wav_data = np.append(wav_data, np.zeros(padding))
            y_labels = np.zeros(len(wav_data))

        if len(wav_data) != len(y_labels):
            print("wav/y_label length mismatch")
            return None

        wavs = np.append(wavs, wav_data)
        labels = np.append(labels, y_labels)

    print("data loaded")
    return (wavs, labels)
def FrameToFeatures(frame_time_domain, sampling_rate):
    frame_time_domain = np.array(frame_time_domain).astype(np.float32)
    # Log frame energy
    log_frame_energy = math.log(max(np.sum(np.square(frame_time_domain)), 0.0000001))

    signal_delayed = frame_time_domain[1:]
    signal_clipped = frame_time_domain[:-1]
    normalized_autocorrelation_lag_1 = np.dot(signal_clipped, signal_delayed) / max(0.001, math.sqrt(np.sum(np.square(signal_clipped)) * np.sum(np.square(signal_delayed))))

    # Bandpassed energy in 80 to 450 Hz
    freq_domain_amplitude = np.abs(np.fft.fft(frame_time_domain))
    lo_index = int((90 / (sampling_rate / 2.0)) * len(frame_time_domain))
    hi_index = int((300 / (sampling_rate / 2.0)) * len(frame_time_domain))
    
    high_pitched_noise_rel = np.sum(freq_domain_amplitude[hi_index:])/max(np.sum(freq_domain_amplitude[lo_index:hi_index]), 0.0000001)

    low_pitched_noise_rel = np.sum(freq_domain_amplitude[:lo_index])/max(np.sum(freq_domain_amplitude[lo_index:hi_index]), 0.0000001)

    relative_energy_in_vocal_range = np.sum(freq_domain_amplitude[lo_index:hi_index]) / max(np.sum(freq_domain_amplitude), 0.0000001)

    return [log_frame_energy, high_pitched_noise_rel, low_pitched_noise_rel, normalized_autocorrelation_lag_1, relative_energy_in_vocal_range]

def LabelledFileToTrainingSamples(sound_data, labels, window_size, stride, sample_rate):
    # Sound data must be rank 1
    # labels must be rank 1
    results_x = []
    results_y = []

    limit = len(sound_data) - window_size
    i = 0
    while i < limit:
        samples = sound_data[i:i + window_size]
        
        
        if (len(samples) < window_size):
            break

        x_element = []
        x_element += FrameToFeatures(samples, sample_rate)
        #previous 4 samples added to data
        if(i-stride>=0):
            samples1 = sound_data[i-stride:i-stride+window_size]
            x_element += FrameToFeatures(samples1, sample_rate)
        else: 
            x_element += [0,0,0,0,0]

        if(i-stride*2>=0):
            samples2 = sound_data[i-stride*2:i-stride*2+window_size]
            x_element += FrameToFeatures(samples2, sample_rate)
        else: 
            x_element += [0,0,0,0,0]
        # if(i-stride*3>=0):
        #     samples3 = sound_data[i-stride*3:i-stride*3+window_size]
        #     x_element += FrameToFeatures(samples3, sample_rate)
        # else: 
        #    x_element += [0,0,0]

        results_x.append(x_element)
        l = labels[i:i + window_size]

        # Classify this time window as vocals
        # Considered vocals is over 20 percent of the time is vocal

        is_speech = np.sum(l) > 0.2

        if (is_speech):
            results_y.append(1)
        else:
            results_y.append(0)
        i += stride

    return results_x, np.expand_dims(np.array(results_y).astype(np.float32), 1)

if __name__ == "__main__":
    x_data, y_data = MergeData(2048)

    print(len(y_data))
    percentage = np.mean(y_data)
    print("voice percentage: " + str(percentage))

    SAMPLE_RATE = 44100
    WINDOW_SIZE = 2048
    STRIDE = int(WINDOW_SIZE / 2)
    x_train, y_train = LabelledFileToTrainingSamples(x_data, np.reshape(y_data, [-1]), WINDOW_SIZE, STRIDE, SAMPLE_RATE)
    
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    print("SIZES!")
    print(len(x_data)/STRIDE)
    print(len(y_train))
    print(np.mean(y_train))

    false_positives = 0
    false_negatives = 0

    vad = VAD("saved_models/model_15_features_test_2.h5")
    j=0
    i=0
    while j<len(y_train):
        if(i>STRIDE*3):
            is_voice = vad.is_voice(x_data[i:i+WINDOW_SIZE],x_data[i-STRIDE:i-STRIDE+WINDOW_SIZE],x_data[i-STRIDE*2:i-STRIDE*2+WINDOW_SIZE])
        else:
            is_voice = False

        if(is_voice == True and y_train[j] == 0):
            false_positives = false_positives + 1
        if(is_voice == False and y_train[j] == 1):
            false_negatives = false_negatives + 1
        #print(i);
        j=j+1
        i=i+STRIDE

    print("RESULTS:")
    print("total error")
    print((false_positives+false_negatives)/(len(y_train)))
    print("false positives")
    print(false_positives/(len(y_train)*np.mean(y_train)))
    print("false negatives")
    print(false_negatives/(len(y_train)*(1-np.mean(y_train))))


