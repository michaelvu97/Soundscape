import numpy as np
import tensorflow as tf
import scipy.io.wavfile
import scipy.misc
import math
import os
import re
import h5py
import matplotlib.pyplot as plt

NUM_HISTORY = 3
NUM_FEATURES = 3

def MergeData(window_size):
    file_list = [
        "./sound_files/room-44100"
        #,"./sound_files/no_speech"
    ]

    wavs = np.zeros(window_size)
    labels = np.zeros(window_size)
    for file in file_list:
        rate, wav_data = scipy.io.wavfile.read(file + ".wav")
        print(file)
        if rate != 44100:
            print("input file does not have 44.1kHz sample rate: " + file)
            return None

        if(file != "./sound_files/no_speech"):
            y_labels = np.load(file + ".npy").astype(np.float32)

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


# def FrameToFeatures(frame_time_domain):
#     frame_length = len(frame_time_domain)
#     frame_time_domain = np.array(frame_time_domain).astype(np.float32)
#     # Accepts a rank 1 np.array [frame_data]

#     log_frame_energy = math.log(max(np.sum(np.square(frame_time_domain)), 0.001))

#     zero_crossing_rate = 0.0
#     for i in range(frame_length - 1):
#         if (frame_time_domain[i] >= 0) != (frame_time_domain[i + 1] >= 0):
#             zero_crossing_rate += 1.0

#     signal_delayed = frame_time_domain[1:]
#     signal_clipped = frame_time_domain[:-1]

#     normalized_autocorrelation_lag_1 = np.dot(signal_clipped, signal_delayed) / max(0.001, math.sqrt(np.sum(np.square(signal_clipped)) * np.sum(np.square(signal_delayed))))

#     poly_degree = 12

#     predictor_coeffs = np.polyfit(range(frame_length), frame_time_domain, poly_degree)
#     first_linear_predictor_coeff = predictor_coeffs[0]

#     # log of MSE, had to guess MSE
#     predictor_err = np.sum(np.square(np.polyval(predictor_coeffs, range(frame_length)) - frame_time_domain))

#     log_linear_predictor_err = math.log(max(predictor_err, 0.001))

#     return [log_frame_energy, zero_crossing_rate, normalized_autocorrelation_lag_1, first_linear_predictor_coeff, log_linear_predictor_err]

def FrameToFeatures(frame_time_domain, sampling_rate):
    frame_time_domain = np.array(frame_time_domain).astype(np.float32)
    # Log frame energy
    log_frame_energy = math.log(max(np.sum(np.square(frame_time_domain)), 0.00001))

    signal_delayed = frame_time_domain[1:]
    signal_clipped = frame_time_domain[:-1]
    normalized_autocorrelation_lag_1 = np.dot(signal_clipped, signal_delayed) / max(0.001, math.sqrt(np.sum(np.square(signal_clipped)) * np.sum(np.square(signal_delayed))))

    # Bandpassed energy in 80 to 450 Hz
    freq_domain_amplitude = np.abs(np.fft.fft(frame_time_domain))
    lo_index = int((80 / (sampling_rate / 2.0)) * len(frame_time_domain))
    hi_index = int((450 / (sampling_rate / 2.0)) * len(frame_time_domain))
    
    relative_energy_in_vocal_range = np.mean(freq_domain_amplitude[lo_index:hi_index]) / max(np.max(freq_domain_amplitude[lo_index:hi_index]), 0.00001)

    return [log_frame_energy, normalized_autocorrelation_lag_1, relative_energy_in_vocal_range]

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

        x_element = np.zeros((3,3))
        x_element[0] = FrameToFeatures(samples, sample_rate)
        #previous 4 samples added to data
        if(i-stride>=0):
            samples1 = sound_data[i-stride:i-stride+window_size]
            x_element[1] = FrameToFeatures(samples1, sample_rate)

        if(i-stride*2>=0):
            samples2 = sound_data[i-stride*2:i-stride*2+window_size]
            x_element[2] = FrameToFeatures(samples2, sample_rate)
        # if(i-stride*3>=0):
        #     samples3 = sound_data[i-stride*3:i-stride*3+window_size]
        #     x_element += FrameToFeatures(samples3, sample_rate)
        # else: 
        #    x_element += [0,0,0]

        x_element = np.reshape(x_element, [-1])
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
    SAMPLE_RATE = 44100
    WINDOW_SIZE = 2048

    x_data, y_data = MergeData(WINDOW_SIZE)

    # Transform to windowed

    print(len(y_data))
    percentage = np.mean(y_data)
    print("voice percentage: " + str(percentage))

    def GetModel():
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(40, activation="relu", kernel_initializer="glorot_normal", input_shape=((NUM_FEATURES*NUM_HISTORY),)))
        model.add(tf.keras.layers.Dense(10, activation="relu", kernel_initializer="glorot_normal"))
        model.add(tf.keras.layers.Dense(1, activation="sigmoid", kernel_initializer="glorot_normal"))
        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
        return model

    model = GetModel()

    STRIDE = int(WINDOW_SIZE / 2)
    x_train, y_train = LabelledFileToTrainingSamples(x_data, np.reshape(y_data, [-1]), WINDOW_SIZE, STRIDE, SAMPLE_RATE)
    # print(x_train)
    print("voice frame percentage: " + str(np.mean(y_train)))

    print("Fitting model on training data")
    print(np.array(x_train).shape)
    print(y_train.shape)
    x_train = np.stack(x_train)
    y_train = np.stack(y_train)
    # class_weight = {0: 1.0, 1: 0.5}
    class_weight = None
    history = model.fit(x_train, y_train, epochs=100, validation_split=0.2, class_weight=class_weight)

    model.save('./saved_models/model_3x9_features.h5')
