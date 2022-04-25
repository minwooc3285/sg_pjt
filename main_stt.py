from ETRI_STT import STT_model
from scipy.io import wavfile
import time

path  = "./"
fname = "test_angry.wav"

fs, data = wavfile.read(path + fname) # Sample rate of wav file, Data read from wav file
print(fs, data.shape)
print(data)

STT = STT_model()

audio_in_bytes = data

print('start infernce')

start = time.time()
text = STT.inference(audio_in_bytes)
end = time.time

print(end-start)
print('output:',text)