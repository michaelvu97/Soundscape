"""
testing out non blocking mode
"""

import pyaudio
import time

WIDTH = 2
CHANNELS = 2
RATE = 44100

def streamCallback(in_data, frame_count, time_info, status):
    print(time_info)
    return (in_data, pyaudio.paContinue)

def in_progress(stream):
    return stream.is_active()

if __name__ == "__main__":

    p = pyaudio.PyAudio()

    stream = p.open(format=p.get_format_from_width(WIDTH),
                channels=CHANNELS,
                rate=RATE,
                input=True,
                stream_callback=callback)

    stream.start_stream()

    while in_progress():
        time.sleep(0.1)

    stream.stop_stream()
    stream.close()

    p.terminate()
