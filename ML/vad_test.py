import tensorflow as tf
import numpy as np
import scipy
from vad import FrameToFeatures
import matplotlib.pyplot as plt
import h5py

class VAD:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)

    def is_voice(self, samples, samples1,samples2):
        features = []
        features += FrameToFeatures(samples, 44100)
        features += FrameToFeatures(samples1, 44100)
        features += FrameToFeatures(samples2, 44100)
        #features += FrameToFeatures(samples3, 44100)
        feat_arr = np.array([features])
        feat_arr.transpose()

        return self.model.predict_proba(feat_arr, verbose=0)[0][0] > 0.4
        # return self.model.predict_classes(features, verbose=0)[0][0] == 1

if __name__ == "__main__":
    rate, wav = scipy.io.wavfile.read("./sample-000080.wav")

    # add noise
    # wav = wav + 25.0 * np.ran dom.normal(size=len(wav))
    wav = wav.astype(np.float32)

    minutes_to_samples = 44100 * 60

    WINDOW_SIZE = 2048 # 0.1 second window
    STRIDE = int(WINDOW_SIZE/2)

    vad = VAD('./saved_models/model_3x9_features.h5')

    results = []

    i = 0
    while i < len(wav):
        if(i>STRIDE*3):
            res = vad.is_voice(wav[i:i+WINDOW_SIZE],wav[i-STRIDE:i-STRIDE+WINDOW_SIZE],wav[i-STRIDE*2:i-STRIDE*2+WINDOW_SIZE])
        else:
            res=0;
        # is_speech = res[0][0] > 0.5

        # results.append(is_speech)
        results.append(res)

        i += STRIDE

    fig, ax1 = plt.subplots()

    ax1.plot(wav, color="red")
    ax2 = ax1.twinx()

    ax2.plot(np.linspace(0, len(wav), len(results)), np.array(results).astype(np.int32))

    plt.show()