import os
import re
files = [x[:-len(".mp3")] for x in os.listdir() if re.search(".mp3$", x)]
print(files)

for file in files:
    cmd = '("C:\\Users\\Michael Vu\\Documents\\FFMPEG\\bin\\ffmpeg.exe"' + " -i " + file + ".mp3 -vn " + file + ".wav)" 
    os.system(cmd)