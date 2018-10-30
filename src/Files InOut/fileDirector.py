
#from __future__ import print_function
import scipy.io.wavfile as wavfile
#import scipy
#import scipy.fftpack
#import numpy as np
#from matplotlib import pyplot as plt



def openWav(fileName):
    try:
        fs_rate, signal = wavfile.read("../Wav/"+fileName+".wav")
        return fs_rate, signal
    except:
        print ("\nError: El archivo de audio "+fileName+".wav"+" no existe\n")
    return []

def openWavToFreqTime(fileName):
    try:
        samplingFrequency, signalData = wavfile.read("../Wav/"+fileName+".wav")
        return samplingFrequency, signalData
    except:
        print ("\nError: El archivo de audio "+fileName+".wav"+" no existe\n")
    return []




