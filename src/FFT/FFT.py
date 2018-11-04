
from __future__ import print_function
import scipy.io.wavfile as wavfile
import scipy
import scipy.fftpack
import numpy as np
from matplotlib import pyplot as plt



def calculateFFT(fs_rate, signal, flag):
    
    print(flag)
    l_audio = len(signal.shape)
    print ("Channels", l_audio)
    if l_audio == 2:
        signal = signal.sum(axis=1) / 2
    N = signal.shape[0]
    print ("Complete Samplings N", N)
    secs = N / float(fs_rate)
    print ("secs", secs)
    Ts = 1.0/fs_rate # sampling interval in time
    print ("Timestep between samples Ts", Ts)
    t = scipy.arange(0, secs, Ts) # time vector as scipy arange field / numpy.ndarray
    FFT = abs(scipy.fft(signal))
    freqs = scipy.fftpack.fftfreq(signal.size, t[1]-t[0])
    if(flag=="1"):
        graphics(FFT, freqs, signal, t,fs_rate)
    return 

def graphics(FFT, freqs, signal, t,fs_rate):
    plt.subplot(311)
    p1 = plt.plot(t, signal, "g") # plotting the signal
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.subplot(312)
    print(FFT)
    p2 = plt.plot(freqs, FFT, "r") # plotting the complete fft spectrum
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Count dbl-sided')
    plt.subplot(313)
    plt.specgram(signal,Fs=fs_rate)
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.show()

def getFrequencyTimePlot(samplingFrequency,signalData):
    # Plot the signal read from wav file
    plt.subplot(211)
    plt.title('Spectrogram of a wav file with piano music')
    plt.plot(signalData)
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.subplot(212)
    plt.specgram(signalData,Fs=samplingFrequency)
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.show()