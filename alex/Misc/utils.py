import pyaudio
p = pyaudio.PyAudio()

def get_device_indices():
    deviceIndices = [];
    USB_DEVICE_NAME = "USB Audio Device"
    for ii in range(p.get_device_count()):
        name = (p.get_device_info_by_index(ii).get('name'))
        print(name)
        if name == USB_DEVICE_NAME:
          deviceIndices.append(ii) 
    return (deviceIndices)


if __name__ == "__main__":
    print(get_device_indices()) 
