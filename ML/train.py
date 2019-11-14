import numpy as np
import tensorflow as tf
import scipy.io.wavfile
import scipy.misc
import math
import os
import re
import matplotlib.pyplot as plt

def get_data():
    files = [x for x in os.listdir() if re.search("sample.+?\.wav$", x)]
    files = files[:10] # For now (on my laptop), only use the first 10 samples
    files_sampled = []
    rate = 0
    for file in files:
        rate, sample = scipy.io.wavfile.read(file)
        files_sampled.append(sample.astype(np.float32))
    return rate, files_sampled


def get_random_background_noise(sample, bg_sample):
    # This assumes that the bg noise wav has the same sample rate
    sample_len = len(sample)
    bg_sample_len = len(bg_sample)
    max_bg_sample_idx = bg_sample_len - sample_len - 1
    start = np.random.randint(0, max_bg_sample_idx)
    return bg_sample[start:start+sample_len]

def normalize(sample):
    max_amplitude = (1 << 15) - 1
    sample_max_amplitude = math.sqrt(max([x ** 2 for x in sample]))
    multiplier = max_amplitude / sample_max_amplitude
    return sample * multiplier

rate, samples = get_data()
np.random.seed(42069)
np.random.shuffle(samples)

# Normalize samples
samples = [normalize(x) for x in samples]

# Add noise to samples
# TODO add noise to tf model instead.
bg_rate, bg_sample = scipy.io.wavfile.read('./BG/BG_COFFEE.wav')
bg_sample = bg_sample.astype(np.float32)[:,0]
samples_with_noise = np.array([x + get_random_background_noise(x, bg_sample) for x in samples])
bg_sample = None

def spectro(x, name):
    # plt.imshow(np.transpose(x))
    # plt.colorbar()
    # plt.show()
    scipy.misc.imsave(name, x)

scipy.io.wavfile.write("TEST_INPUT_WITH_NOISE.wav", rate, np.concatenate(samples_with_noise).astype(np.int16))

window_size_seconds = 0.05
window_size_samples = int(window_size_seconds * rate)
frame_step = int(window_size_samples / 4)

MODEL_INPUT_TIME_SECONDS = 0.2 # 0.1 second sample
MODEL_INPUT_TIME_SAMPLES = int(MODEL_INPUT_TIME_SECONDS * rate)

input_dims = [
    None, # Batch
    MODEL_INPUT_TIME_SAMPLES
]

x_time_domain = tf.placeholder(np.float32, input_dims)
y_true_time_domain = tf.placeholder(np.float32, input_dims)
y_true_freq_domain = tf.contrib.signal.stft(y_true_time_domain, window_size_samples, frame_step)

x_freq_domain = tf.contrib.signal.stft(x_time_domain, window_size_samples, frame_step)

num_frequencies = int(x_freq_domain.shape[2])
num_stft_windows = int(x_freq_domain.shape[1])

x_freq_flattened = tf.reshape(x_freq_domain, [-1, num_frequencies * num_stft_windows])

# Network
x_freq_domain_to_real = tf.expand_dims(tf.abs(x_freq_domain), 3)

hidden = tf.keras.layers.Conv2D(96, kernel_size=11, strides=4, padding="same", activation='relu', input_shape=(num_stft_windows, num_frequencies, 1))(x_freq_domain_to_real)
hidden = tf.keras.layers.Conv2D(256, kernel_size=5, padding="same", activation="relu")(hidden)
hidden = tf.keras.layers.MaxPooling2D(pool_size=2)(hidden)
hidden = tf.keras.layers.Conv2D(300, kernel_size=3, padding="same", activation="relu")(hidden)
hidden = tf.keras.layers.Conv2D(300, kernel_size=3, padding="same", activation="relu")(hidden)
hidden = tf.keras.layers.MaxPooling2D(pool_size=2)(hidden)
# hidden = tf.keras.layers.Dropout(0.25)(hidden)

hidden = tf.keras.layers.Flatten()(hidden)
# hidden = tf.keras.layers.Dropout(0.25)(hidden)
hidden = tf.keras.layers.Dense(4096, activation="relu")(hidden)
hidden = tf.keras.layers.Dense(4096, activation="relu")(hidden)

y_hat = tf.keras.layers.Dense(num_frequencies * num_stft_windows, activation="linear")(hidden)

confidence = tf.constant(0.5, tf.float32)

result_masked_limited = tf.multiply(x_freq_flattened, tf.cast(tf.greater(y_hat, confidence), tf.complex64))

result_masked_limited_unflattened = tf.reshape(result_masked_limited, [-1, num_stft_windows, num_frequencies])
reconstructed = tf.contrib.signal.inverse_stft(result_masked_limited_unflattened, window_size_samples, frame_step)

# This will have to be automated using SNR or SAR or something.
y_signal_binary_threshold = tf.constant(20000, tf.float32)
y_ideal_binary_mask = tf.cast(tf.greater(tf.abs(y_true_freq_domain), y_signal_binary_threshold), tf.float32)

error_coefficients = tf.constant(np.sqrt(np.linspace(1, 0.01, num_frequencies)), tf.float32)

error_gradient = tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.reshape(y_hat, [-1, 1]), labels=tf.reshape(y_ideal_binary_mask, [-1, 1]))
error_low_pass = tf.square(tf.sigmoid(tf.reshape(y_hat,[-1, num_stft_windows, num_frequencies])) - error_coefficients)
err = tf.reduce_mean(error_gradient) + 0.25 * tf.reduce_mean(error_low_pass)

optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(err)

def SplitIntoNSizedWindows(arrs, n):
    # arrs is a list of sound files
    result = []

    for arr in arrs: 
        clip_length = len(arr) % n
        arr = arr[:-clip_length]
        num_partitions = len(arr) / n

        if num_partitions <= 0:
            continue

        split_arr = np.split(arr, num_partitions)
        for partition in split_arr:
            result.append(partition)

    return np.vstack(result)

sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
sess.run(init)

data_dict = {
    y_true_time_domain: SplitIntoNSizedWindows(samples[:1], MODEL_INPUT_TIME_SAMPLES), 
    x_time_domain: SplitIntoNSizedWindows(samples_with_noise[:1], MODEL_INPUT_TIME_SAMPLES)
}

test_dict = {
    y_true_time_domain: SplitIntoNSizedWindows(samples[9:], MODEL_INPUT_TIME_SAMPLES),
    x_time_domain: SplitIntoNSizedWindows(samples[9:], MODEL_INPUT_TIME_SAMPLES)
}

r = sess.run(y_ideal_binary_mask, feed_dict=data_dict)
spectro(np.reshape(r, [-1, num_frequencies]), "ideal_mask.png")
print("spectro'd")

print("frequencies: " + str(num_frequencies))
print("stft_windows: " + str(num_stft_windows))


for i in range(15):
    sess.run(optimizer, feed_dict=data_dict)
    train_err = sess.run(err, feed_dict=data_dict)
    test_err = sess.run(err, feed_dict=test_dict)
    curr_mask_train = sess.run(tf.cast(tf.greater(y_hat, confidence), tf.int16), feed_dict=data_dict)
    curr_mask_test = sess.run(tf.cast(tf.greater(y_hat, confidence), tf.int16), feed_dict=test_dict)
    spectro(np.reshape(curr_mask_train, [-1, num_frequencies]), "./test_masks/" + str(i) + "-train.png")
    spectro(np.reshape(curr_mask_test, [-1, num_frequencies]), "./test_masks/" + str(i) + "-test.png")
    print(str(i) +": " + str(train_err) + "/" + str(test_err))

# Generate some test audio?
test = sess.run(reconstructed, feed_dict=data_dict)

scipy.io.wavfile.write("TRAIN.wav", rate, test.flatten().astype(np.int16))
scipy.io.wavfile.write("TEST.wav", rate, sess.run(reconstructed, feed_dict=test_dict).flatten().astype(np.int16))

spectro(np.reshape(sess.run(tf.abs(result_masked_limited), feed_dict=data_dict), [-1, num_frequencies]), "result_masked.png")
spectro(np.reshape(sess.run(tf.cast(tf.greater(y_hat, confidence), tf.int16), feed_dict=data_dict),[-1, num_frequencies]), "result_mask.png")
spectro(np.reshape(sess.run(tf.abs(x_freq_flattened), feed_dict=data_dict),[-1, num_frequencies]), "input.png")



# plt.plot(errs)
# plt.show()
