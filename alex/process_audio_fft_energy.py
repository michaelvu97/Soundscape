#!/Library/Frameworks/Python.framework/Versions/3.6/bin/python3
import pyaudio
import pygame
import wave
import struct
import time
import numpy as np
from scipy.fftpack import fft

form_1 = pyaudio.paInt8 # 16-bit resolution
chans = 2 # 1 channel
samp_rate = 44100 # 44.1kHz sampling rate
chunk = 4096 # 2^12 samples for buffer
record_secs = 10 # seconds to record
dev_index = 3 # device index found by p.get_device_info_by_index(ii)
frequencies = []
#sources = 200
sources = 100
max_energy = 0;
min_energy = 1;
noise_gate = 1e-10

# Define the colors we will use in RGB format
BLACK = (  0,   0,   0)
WHITE = (255, 255, 255)
BLUE =  (  0,   0, 255)
GREEN = (  0, 255,   0)
RED =   (255,   0,   0)
audio = pyaudio.PyAudio() # create pyaudio instantiation

# Set the height and width of the screen
height = 600
width = 1000
size = [width, height]
screen = pygame.display.set_mode(size)
 
# Pre fill an array with the frequency data  corresponding to each index of the fft
for i in range(int(chunk/2)):
	temp = int(samp_rate * i / chunk)
	frequencies.append(temp)

# create pyaudio stream
stream = audio.open(format = form_1,rate = samp_rate,channels = chans, \
                    input_device_index = dev_index,input = True, \
                    frames_per_buffer=chunk)
#print("recording")
frames = []
done = False
horiz = 0
# do this forever until program quits
while not done:

	for event in pygame.event.get(): # User did something
		if event.type == pygame.QUIT: # If user clicked close
			done=True # Flag that we are done so we exit this loop
 
# Read the stream data into a variable
	data = stream.read(chunk, exception_on_overflow=False)

# Convert the stream data into an integer array
	data_int = np.fromstring(data, dtype=np.int8)

# Get the data for both channels 1 and 2
	chan1 = np.array(data_int[::2])
	chan2 = np.array(data_int[1::2])

# Compute the faurier transform of each channel
	chan1_fft_temp = fft(chan1)
	chan2_fft_temp = fft(chan2)

	chan1_fft = np.abs(chan1_fft_temp[0:chunk]) * 2 / (256 * chunk)
	chan2_fft = np.abs(chan2_fft_temp[0:chunk]) * 2 / (256 * chunk)


# Compute the enery in the fft from each channel
	chan1_energies = []
	chan2_energies = []

	for i in range(sources):
		chan1_energies.append(0)
		chan2_energies.append(0)

	#for i in range(len(chan1_fft)):
	#	chan1_energy = chan1_energy + chan1_fft[i]*chan1_fft[i]
	#	chan2_energy = chan2_energy + chan2_fft[i]*chan2_fft[i]

	for i in range(sources):
		start = int(i * (chunk / 2) / (sources))
		end = int((i + 1) * (chunk / 2) / (sources))

		#print("Start: ", start)
		#print("End:   ", end)

		for j in range(start, end):
			chan1_energies[i] = chan1_energies[i] + chan1_fft[j]*chan1_fft[j]
			chan2_energies[i] = chan2_energies[i] + chan2_fft[j]*chan2_fft[j]

		chan1_energies[i] = chan1_energies[i] / chunk
		chan2_energies[i] = chan2_energies[i] / chunk

	
# Get a measure of which channel the audio is coming from. Positive means 1, negative means 2
	#direction = (chan1_energy - chan2_energy) / (chan1_energy + chan2_energy)
	directions = [0] * sources
	screen.fill(WHITE)
	for i in range(sources):
		directions[i] = (chan1_energies[i] - chan2_energies[i]) / (chan1_energies[i] + chan2_energies[i])

		if (chan1_energies[i] + chan2_energies[i]) < noise_gate:
			directions[i] = 0

		#print("Energy: ", chan1_energies[i] + chan2_energies[i], " 1: ", chan1_energies[i], " 2: ", chan2_energies[i])

		horiz = directions[i] * width/2 + width / 2 - 25
		#print(horiz)
		#print()

	#screen.fill(WHITE)

		R = int(255 * i / sources)
		G = (R + 30) % 256
		B = (G + 85) % 256
		pygame.draw.rect(screen, (R, G, B), [horiz, int((height/sources) * i), 50, int(height/sources)])
	pygame.display.flip()

pygame.quit()

#print("finished recording")

# stop the stream, close it, and terminate the pyaudio instantiation
stream.stop_stream()
stream.close()
audio.terminate()

