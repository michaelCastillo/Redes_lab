
import scipy.io.wavfile as wavfile

def writeWav(fileName, rate, data):
    wavfile.write(fileName, rate, data)
   