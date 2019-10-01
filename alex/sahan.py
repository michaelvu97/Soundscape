#!/usr/bin/python
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
audio1 = pyaudio.PyAudio()

# Set the height and width of the screen
height = 600
width = 1000
size = [width, height]

def initializeFrequencyArray(fArray, chan1_energies, chan2_energies):
    # Pre fill an array with the frequency data  corresponding to each index of the fft
    for i in range(int(chunk/2)):
        temp = int(samp_rate * i / chunk)
        frequencies.append(temp)

def getChannelEnergies(data, numChannels):
    # Convert the stream data into an integer array
    data_int = np.fromstring(data, dtype=np.int8)
        
    # Get the data for n channels
    channels = []
    for i in range(numChannels):
        channels[i] = np.array(data_int[i::n])
    
    # Compute the faurier transform of each channel
    fftOfChannels = []
    for i in range(numChannels):
        channel = fft(channels[i])
        ###########################################watch out
        channel = np.abs(channel[0:chunk]) * 2 / (256 * chunk)
        fftOfChannels[i] = channel

    # Compute the enery in the fft from each channel
    channelEnergies =[]
    for j in range(numChannels):
        channelEnergies[j] = []
        for i in range(sources):
            channelEnergies[i].append(0)


	for i in range(sources):
	    start = int(i * (chunk / 2) / (sources))
	    end = int((i + 1) * (chunk / 2) / (sources))
        
        for cNum in range(numChannels):
            channelEnergy = channelEnergies[cNum]
            channelFft = fftOfChannels[cNum]
            for j in range(start, end):
	            #approximation of energy stored in section fft
                #ask mert
                channelEnergy[i] = channelEnergy[i] + channelFft[j]*channelFft[j]
	
	        channelEnergy[i] = channelEnergy[i] / chunk
	
    return channelEnergies

def updateScreenWithEnergies(chan1_energies, chan2_energies):
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

	
if __name__ == "__main__":	
	screen = pygame.display.set_mode(size)
	initializeFrequencyArray(frequencies)
	
	
	
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
	    	
	        channelEnergies = getChannelEnergies(data)
	        updateScreenWithEnergies(channelEnergies[0], channelEnergies[1])
	pygame.quit()
	
	#print("finished recording")
	
	# stop the stream, close it, and terminate the pyaudio instantiation
	stream.stop_stream()
	stream.close()
	audio.terminate()
	
