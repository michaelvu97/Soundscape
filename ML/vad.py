import numpy as np
import tensorflow as tf
import scipy.io.wavfile
import scipy.misc
import math
import os
import re
import matplotlib.pyplot as plt

NUM_FEATURES = 5

def FrameToFeatures(frame_time_domain):
    frame_length = len(frame_time_domain)
    frame_time_domain = np.array(frame_time_domain).astype(np.float32)
    # Accepts a rank 1 np.array [frame_data]

    log_frame_energy = math.log(np.sum(np.square(frame_time_domain)))

    zero_crossing_rate = 0.0
    for i in range(frame_length - 1):
        if (frame_time_domain[i] >= 0) != (frame_time_domain[i + 1] >= 0):
            zero_crossing_rate += 1.0


    signal_delayed = frame_time_domain[1:]
    signal_clipped = frame_time_domain[:-1]

    normalized_autocorrelation_lag_1 = np.dot(signal_clipped, signal_delayed) / math.sqrt(np.sum(np.square(signal_clipped)) * np.sum(np.square(signal_delayed)))

    poly_degree = 12

    predictor_coeffs = np.polyfit(range(frame_length), frame_time_domain, poly_degree)
    first_linear_predictor_coeff = predictor_coeffs[0]

    # log of MSE, had to guess MSE
    predictor_err = np.sum(np.square(np.polyval(predictor_coeffs, range(frame_length)) - frame_time_domain))

    log_linear_predictor_err = math.log(predictor_err)

    return [log_frame_energy, zero_crossing_rate, normalized_autocorrelation_lag_1, first_linear_predictor_coeff, log_linear_predictor_err]

def LabelledFileToTrainingSamples(sound_data, labels, window_size):
    results_x = []
    results_y = []

    limit = len(sound_data) - window_size
    for i in range(limit):
        samples = sound_data[i:i + window_size]
        labels = labels[i:i + window_size]

        # Classify this time window as vocals
        # Considered vocals is over 20 percent of the time is vocals
        is_speech = np.mean(labels) > 0.2

        results_x.append(samples)
        results_y.append(is_speech)

    return results_x, results_y


rate, wav_data = scipy.io.wavfile.read("room-44100.wav")
y_labels = np.load("room.npy")

print("data loaded")

# Fix off-by-one errors
min_length = min(len(y_labels), len(wav_data))
wav_data = wav_data[:min_length].astype(np.float32)
y_labels = y_labels[:min_length]

# Transform to windowed
WINDOW_SIZE = 4410 # 0.1 second window
x_data_windows, y_data_windows = LabelledFileToTrainingSamples(wav_data, y_labels, WINDOW_SIZE)

print("data windowed")

num_windows = len(y_data_windows)

x_train = [FrameToFeatures(x) for x in x_data_windows]
y_train = y_data_windows

print("features extracted")

def GetModel():
    inputs = tf.keras.Input(shape=(NUM_FEATURES,), name="features")
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="predictions")(inputs)
    return tf.keras.models.Model(inputs=inputs, outputs=outputs)

model = GetModel()
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

print("Fitting model on training data")
history = model.fit(x_train, y_train, epochs=500, validation_split=0.1)

print("History= " + str(history))