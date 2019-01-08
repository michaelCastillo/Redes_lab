
import scipy.io.wavfile as wavfile
import wave
import struct
import numpy as np
def writeWavPerFrame(fileName, rate, data):
    frames=rate
    
    comptype="NONE"
    
    compname="not compressed"
    
    nchannels=1
    
    sampwidth=2
    wav_file=wave.open("../../Wav/"+fileName+".wav", 'w')
    wav_file.setparams((nchannels, sampwidth, int(rate), len(data), comptype, compname))
    for a in data:
        wav_file.writeframes(a)
def writeWav(fileName, rate, data):
    #data=data.astype(np.int32)
    #print("Señal a escribir: ",data)
    wavfile.write("../../Wav/"+fileName+".wav", rate, data)
def linkWavFiles(threads, title):
        i = 0
        y = []
        while i<threads:
                fs, signal = wavfile.read("../../Wav/"+title+str(i+1)+".wav")
                y.extend(signal)
        wavfile.write("../../Wav/"+title+"Final"+".wav", fs, y)