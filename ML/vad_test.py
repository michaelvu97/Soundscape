import tensorflow as tf
import numpy as np
import scipy
from vad import FrameToFeatures
import matplotlib.pyplot as plt
import h5py
import keras

class VAD:
    def __init__(self, model_path):
        self.model = keras.models.load_model(model_path)
        self.triggered = False
        self.decay_factor = 0.1

    def is_voice_vectorized(self, samples, samples1, samples2, smoothing=True):
        return np.array(
            [
                self.is_voice(samples[i], samples1[i], samples2[i]) for i in range(samples.shape[0]) 
            ]
        )

    def is_voice(self, samples, samples1,samples2, smoothing=True):
        features = []
        features += FrameToFeatures(samples, 44100)
        features += FrameToFeatures(samples1, 44100)
        features += FrameToFeatures(samples2, 44100)
        #features += FrameToFeatures(samples3, 44100)
        feat_arr = np.array([features])
        feat_arr.transpose()

        threshold_trigger_on = 0.6
        threshold_trigger_off = 0.5

        new_prob = self.model.predict_proba(feat_arr, verbose=0)[0][0]
        # printProbability(new_prob)
        if smoothing and self.triggered:
            if new_prob < threshold_trigger_off:
                self.triggered = False
                return False
            return True
        if smoothing and not self.triggered:
            if new_prob > threshold_trigger_on:
                self.triggered = True
                return True
            return False    

        return new_prob > threshold_trigger_on

def printProbability(prob):
    normalized = int(prob * 10)
    res = "["
    for i in range(normalized):
        res += "="
    for i in range(10 - normalized):
        res += " "
    res += "]"
    print(res)

if __name__ == "__main__":
    rate, wav = scipy.io.wavfile.read("./sample-000080.wav")

    # add noise
    # wav = wav + 25.0 * np.ran dom.normal(size=len(wav))
    wav = wav.astype(np.float32)

    minutes_to_samples = 44100 * 60

    WINDOW_SIZE = 2048 # 0.1 second window
    STRIDE = int(WINDOW_SIZE/2)

    vad = VAD('./saved_models/model_15_features_test.h5')

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