import pyaudio
import numpy as np
from device_local import SelectDeviceIndices
"""
device_local.py is a python file you should have locally, but should not commit.
It exports SelectDevicesIndices(DeviceInfo[]) => [A_index, B_index, C_index, 
    D_index]
"""

p = pyaudio.PyAudio()

def get_device_indices():
    deviceIndices = [p.get_device_info_by_index(i) for i in range(p.get_device_count())]
    return SelectDeviceIndices(deviceIndices)

def compute_energy(data_channels):
    if data_channels.ndim == 2:
        return np.sum(np.square(data_channels), axis=1, keepdims=True)
    else:
        return np.sum(np.square(data_channels))

if __name__ == "__main__":
    print([p.get_device_info_by_index(i)["name"] for i in range(p.get_device_count())])
    print(get_device_indices()) 
