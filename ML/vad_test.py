import tensorflow as tf
import numpy as np
import scipy
from vad import FrameToFeatures
import matplotlib.pyplot as plt
import h5py

rate, wav = scipy.io.wavfile.read("./sample-000025.wav")

# add noise
wav = wav + 25.0 * np.random.normal(size=len(wav))

minutes_to_samples = 44100 * 60

WINDOW_SIZE = 4410 # 0.1 second window
STRIDE = 2205

new_model = tf.keras.models.load_model('./saved_models/model.h5')

results = []

i = 0
while i < len(wav):
    features = np.array([FrameToFeatures(wav[i:i+WINDOW_SIZE])])
    res = new_model.predict_classes(features)
    # is_speech = res[0][0] > 0.5

    # results.append(is_speech)
    results.append(res[0][0])

    i += STRIDE

fig, ax1 = plt.subplots()

ax1.plot(wav, color="red")
ax2 = ax1.twinx()

ax2.plot(np.linspace(0, len(wav), len(results)), np.array(results).astype(np.int32))

plt.show()