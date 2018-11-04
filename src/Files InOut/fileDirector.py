
#from __future__ import print_function
import scipy.io.wavfile as wavfile

#import scipy
#import scipy.fftpack
import numpy as np
#from matplotlib import pyplot as plt



def openWav(fileName):
    try:
        fs_rate, signal = wavfile.read("../Wav/"+fileName+".wav")
        fs_signal = []
        print("Signal TIPO")
        print(type(signal[0]))
        if(type(signal[0]) is np.ndarray ):
            print("Es lista")
            for el in signal:
                fs_signal.append(el[0])
            signal = np.asarray(fs_signal)
        print(fs_rate,signal)
        return fs_rate, signal
    except:
        print ("\nError: El archivo de audio "+fileName+".wav"+" no existe\n")
    return []

def openWavToFreqTime(fileName):
    try:
        samplingFrequency, signalData = wavfile.read("../Wav/"+fileName+".wav")
        print("FREQ ")
        print (samplingFrequency,signalData)
        return samplingFrequency, signalData
    except:
        print ("\nError: El archivo de audio "+fileName+".wav"+" no existe\n")
    return []



