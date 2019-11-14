import numpy as np
import tensorflow as tf
import scipy.io.wavfile
import scipy.misc
import math
import os
import re
import h5py
import matplotlib.pyplot as plt

NUM_FEATURES = 5

def FrameToFeatures(frame_time_domain):
    frame_length = len(frame_time_domain)
    frame_time_domain = np.array(frame_time_domain).astype(np.float32)
    # Accepts a rank 1 np.array [frame_data]

    log_frame_energy = math.log(max(np.sum(np.square(frame_time_domain)), 0.001))

    zero_crossing_rate = 0.0
    for i in range(frame_length - 1):
        if (frame_time_domain[i] >= 0) != (frame_time_domain[i + 1] >= 0):
            zero_crossing_rate += 1.0

    signal_delayed = frame_time_domain[1:]
    signal_clipped = frame_time_domain[:-1]

    normalized_autocorrelation_lag_1 = np.dot(signal_clipped, signal_delayed) / max(0.001, math.sqrt(np.sum(np.square(signal_clipped)) * np.sum(np.square(signal_delayed))))

    poly_degree = 12

    predictor_coeffs = np.polyfit(range(frame_length), frame_time_domain, poly_degree)
    first_linear_predictor_coeff = predictor_coeffs[0]

    # log of MSE, had to guess MSE
    predictor_err = np.sum(np.square(np.polyval(predictor_coeffs, range(frame_length)) - frame_time_domain))

    log_linear_predictor_err = math.log(max(predictor_err, 0.001))

    return [log_frame_energy, zero_crossing_rate, normalized_autocorrelation_lag_1, first_linear_predictor_coeff, log_linear_predictor_err]

def LabelledFileToTrainingSamples(sound_data, labels, window_size, stride):
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
        l = labels[i:i + window_size]

        # Classify this time window as vocals
        # Considered vocals is over 20 percent of the time is vocals
        is_speech = np.sum(l) > 0.2

        results_x.append(FrameToFeatures(samples))
        if (is_speech):
            results_y.append(1)
        else:
            results_y.append(0)
        i += stride

    return results_x, np.expand_dims(np.array(results_y).astype(np.float32), 1)

if __name__ == "__main__":
    rate, wav_data = scipy.io.wavfile.read("room-44100.wav")
    y_labels = np.load("room.npy").astype(np.float32)

    print("data loaded")

    # Fix off-by-one errors
    min_length = min(len(y_labels), len(wav_data))
    wav_data = wav_data[:min_length].astype(np.float32)

    # add some noise to the input
    wav_data = wav_data + 25.0 * np.random.normal(size=min_length)

    y_labels = y_labels[:min_length]

    # Take the first n minutes of data
    # start = 44100 * 60 * 20
    # end = 44100 * 60 * 22
    # wav_data = wav_data[start:end]
    # y_labels = y_labels[start:end]

    # Transform to windowed

    print(len(y_labels))
    percentage = np.mean(y_labels)
    print("voice percentage: " + str(percentage))
    
    def GetModel():
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(10, activation="relu", kernel_initializer="glorot_normal", input_shape=(NUM_FEATURES,)))
        model.add(tf.keras.layers.Dense(1, activation="sigmoid", kernel_initializer="glorot_normal"))
        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
        return model

    model = GetModel()

    WINDOW_SIZE = 4410 # 0.1 second window
    STRIDE = 2205
    x_train, y_train = LabelledFileToTrainingSamples(wav_data, np.reshape(y_labels, [-1]), WINDOW_SIZE, STRIDE)

    print(np.mean(y_train))

    print("Fitting model on training data")
    history = model.fit(x_train, y_train, epochs=100, validation_split=0.2)

    model.save('./saved_models/model.h5')
