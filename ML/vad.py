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

dummy_x_train = [[1,2,-1,2],[1,2,3,4,5,3,3,2]]
dummy_y_train = [[1],[0]]

x_train = [FrameToFeatures(x) for x in dummy_x_train]
y_train = dummy_y_train

def GetModel():
    inputs = tf.keras.Input(shape=(NUM_FEATURES,), name="features")
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="predictions")(inputs)
    return tf.keras.models.Model(inputs=inputs, outputs=outputs)

model = GetModel()
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

print("Fitting model on training data")
history = model.fit(x_train, y_train, epochs=500)

print("History= " + str(history))