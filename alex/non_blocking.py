"""
testing out non blocking mode
"""

import pyaudio
import time
from utils import get_device_indices

FORMAT  = pyaudio.paInt16 # 16-bit resolution
CHANNELS = 1
RATE = 44100
CHUNK = 4096

"""
actual callback logic of stream callback

n - name of stream
in_data - recorded data if input=True for the stream
frame_count - number of frames for the data
time_info - dictionary of 
status - PaCallbackFlags
"""
def streamCallback(n, in_data, frame_count, time_info, status):
    print("stream " +  str(n))
    print(time_info)
    return (in_data, pyaudio.paContinue)

"""
generate a callback for some stream
n - name of stream
"""
def streamNCallback(n):
    return ( lambda in_data, frame_count, time_info, status : 
        streamCallback(n, in_data, frame_count, time_info, status)
    )

"""
check to see if all the streams are active

streams - array of pyaudio stream
"""
def in_progress(streams):
    is_active = True;
    for stream in streams:
        is_active = is_active & stream.is_active()
    return is_active;

def start_streams(streams):
    for stream in streams:
        stream.start_stream()

def end_streams(streams):
    for stream in streams:
        stream.stop_stream()
        stream.close()
"""
p -  the instance of pyaudio
index - usb port index
stream_num - the name of the stream
"""
def generate_stream(p, index, stream_num):
    return  p.open(format=FORMAT,
                input_device_index=index,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                stream_callback=streamNCallback(stream_num))

if __name__ == "__main__":

    p = pyaudio.PyAudio()
    indices = get_device_indices();
    streams = [];
    
    #the dev indices and sream numbers look off, but the microphones are paired up like this
    stream1 = generate_stream(p, indices[0], 1)
    stream4 = generate_stream(p, indices[1], 4)     

    streams.append(stream1)
    streams.append(stream4)    

    start_streams(streams)

    while in_progress(streams):
        time.sleep(0.1)
    
    end_streams(streams)

    p.terminate()
