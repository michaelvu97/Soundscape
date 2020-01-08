import pyaudio
p = pyaudio.PyAudio()
for ii in range(p.get_device_count()):
	print(p.get_device_info_by_index(ii).get('name'))

deviceIndices = [p.get_device_info_by_index(i) for i in range(p.get_device_count())]
print(deviceIndices)
