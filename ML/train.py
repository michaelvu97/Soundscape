import numpy as np
import tensorflow as tf
import scipy.io.wavfile
import scipy.misc
import math
import matplotlib.pyplot as plt

rate, sample1 = scipy.io.wavfile.read("sample-000000.wav")
sample1 = sample1.astype(np.float32)

def generatesine(sample_period, length):
    return np.array([math.sin(float(x) * 2 * 3.14 / sample_period) for x in range(length)])


def spectro(x, name):
    # plt.imshow(np.transpose(x))
    # plt.colorbar()
    # plt.show()
    scipy.misc.imsave(name, x)

sample_with_noise = sample1 \
    + 2000.0 * generatesine(rate / (3*440), sample1.shape[0]) #\
    # + np.random.normal(0, 1000, size=(sample1.shape[0])).astype(np.float32)
scipy.io.wavfile.write("TEST_INPUT_WITH_NOISE.wav", rate, sample_with_noise.astype(np.int16))

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
hidden = tf.keras.layers.Conv2D(384, kernel_size=3, padding="same", activation="relu")(hidden)
hidden = tf.keras.layers.Conv2D(384, kernel_size=3, padding="same", activation="relu")(hidden)
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

err = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.reshape(y_hat, [-1, 1]), labels=tf.reshape(y_ideal_binary_mask, [-1, 1])))

optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(err)

def SplitIntoNSizedWindows(arr, n):
    clip_length = len(arr) % n
    arr = arr[:-clip_length]
    num_partitions = len(arr) / n

    split_arr = np.split(arr, num_partitions)
    return np.vstack(split_arr)

sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
sess.run(init)

data_dict = {
    y_true_time_domain: SplitIntoNSizedWindows(sample1, MODEL_INPUT_TIME_SAMPLES), 
    x_time_domain: SplitIntoNSizedWindows(sample_with_noise, MODEL_INPUT_TIME_SAMPLES)
}

r = sess.run(y_ideal_binary_mask, feed_dict=data_dict)
spectro(np.reshape(r, [-1, num_frequencies]), "ideal_mask.png")
print("spectro'd")

print("frequencies: " + str(num_frequencies))
print("stft_windows: " + str(num_stft_windows))

errs = []

for i in range(100):
    sess.run(optimizer, feed_dict=data_dict)
    e = sess.run(err, feed_dict=data_dict)
    errs.append(e)
    curr_mask = sess.run(tf.cast(tf.greater(y_hat, confidence), tf.int16), feed_dict=data_dict)
    spectro(np.reshape(curr_mask, [-1, num_frequencies]), "./test_masks/" + str(i) + ".png")
    print(str(i) +": " + str(e))

# Generate some test audio?
test = sess.run(reconstructed, feed_dict=data_dict)

# Energy
scipy.io.wavfile.write("TEST.wav", rate, test.flatten().astype(np.int16))

spectro(np.reshape(sess.run(tf.abs(result_masked_limited), feed_dict=data_dict), [-1, num_frequencies]), "result_masked.png")
spectro(np.reshape(sess.run(tf.cast(tf.greater(y_hat, confidence), tf.int16), feed_dict=data_dict),[-1, num_frequencies]), "result_mask.png")
spectro(np.reshape(sess.run(tf.abs(x_freq_flattened), feed_dict=data_dict),[-1, num_frequencies]), "input.png")

# plt.plot(errs)
# plt.show()
