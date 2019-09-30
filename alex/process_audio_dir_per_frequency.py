#!/Library/Frameworks/Python.framework/Versions/3.6/bin/python3
import pyaudio
import wave
import struct
import time
import numpy as np
from scipy.fftpack import fft

def fake_main():
	form_1 = pyaudio.paInt8 # 16-bit resolution
	chans = 2 # 1 channel
	samp_rate = 44100 # 44.1kHz sampling rate
	record_secs = 10 # seconds to record
	dev_index = 3 # device index found by p.get_device_info_by_index(ii)
	frequencies = []
	
	audio = pyaudio.PyAudio() # create pyaudio instantiation
	
	# Pre fill an array with the frequency data  corresponding to each index of the fft
	for i in range(int(chunk/2)):
		temp = int(samp_rate * i / chunk)
		frequencies.append(temp)
	
	# create pyaudio stream
	stream = audio.open(format = form_1,rate = samp_rate,channels = chans, \
	                    input_device_index = dev_index,input = True, \
	                    frames_per_buffer=chunk)
	print("recording")
	frames = []
	
	
	# do this forever until program quits
	while True:
	    # Read the stream data into a variable
            data = stream.read(chunk, exception_on_overflow=False)

	# stop the stream, close it, and terminate the pyaudio instantiation
	stream.stop_stream()
	stream.close()
	audio.terminate()

def process_audio_dir(data):
	chunk = 4096 # 2^12 samples for buffer
    	
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
	
	# Get the highest frequency from each channel
		chan1_ind = 0
		chan2_ind = 0
		chan1_max = 0
		chan2_max = 0
		for i in range(0, len(chan1_fft)):
			if chan1_fft[i] > chan1_max:
				chan1_max = chan1_fft[i]
				chan1_ind = i
	
	
			if chan2_fft[i] > chan2_max:
				chan2_max = chan2_fft[i]
				chan2_ind = i
	
	
	#ind 18 and 81 
	#	print("Dominant channel 1 frequency: ", frequencies[chan1_ind], chan1_ind)
	#	print("Dominant channel 2 frequency: ", frequencies[chan2_ind])
	#	if chan1_max > chan2_max:
	#		print("Channel 1 is dominant")
	#	elif chan2_max > chan1_max:
	#		print("Channel 2 is dominant")
		if chan1_fft[18] > chan2_fft[18]:
			print("Channel 1 is dominant for frequency 193")
		elif chan2_fft[18] > chan1_fft[18]:
			print("Channel 2 is dominant for frequency 193")
		if chan1_fft[81] > chan2_fft[81]:
			print("Channel 1 is dominant for frequency 869")
		elif chan2_fft[81] > chan1_fft[81]:
			print("Channel 2 is dominant for frequency 869")
		print()
		print()
		print()
	
	
	#	print("Channel 1: ", chan1_fft)
	#	print("Channel 2: ", chan2)
	#	print()
	#	time.sleep(100)
	
	
	


if __name__ == "__main__":
    fake_main()
