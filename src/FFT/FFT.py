
from __future__ import print_function
import scipy.io.wavfile as wavfile
import scipy
from scipy.signal import butter, lfilter, freqz, firwin
import scipy.fftpack
import numpy as np
from matplotlib import pyplot as plt



#################################
# Calculo y gráfica de la transformada de fourier
################################

def calcFFT(fs_rate,signal):
    fft = scipy.fft(signal)
    fftNormalized = fft/len(signal)    
    xfft = np.fft.fftfreq(len(fftNormalized),1/fs_rate)
    return xfft, fftNormalized

def plotTransform(xfft,fft):
    plt.title("Transformada ")
    plt.xlabel("Frecuency [Hz]")
    plt.ylabel("Amplitude [dB]")
    plt.plot(xfft,abs(fft))
    plt.show()

###################################
# Calculo de paso bajo
###################################

def lowFilter(fs,cutoff,signal):
    nyquist = fs/2
    lowcut = cutoff
    lowcut2 = lowcut/nyquist
    numtaps = cutoff + 1
    filter = firwin(numtaps,cutoff = lowcut2, window = 'hamming' )
    filtered = lfilter(filter,1.0,signal)
    len_signal = len(signal)#obtengo el largo de la señal
    len_signal = float(len_signal)
    time = float(len_signal)/float(fs)#genero el tiempo total del audio
    x = np.arange(0,time,1.0/float(fs))
    return x,filtered

##################################
# Grafica de señal vs tiempo
##################################

def plotSignalTime(signal,t):
    # if(cplot%3 == 0):
    #     cplot = 101 +cplot
    #     plt.subplot(cplot)
    #     plt.figure(cplot)
    # else:
    #     cplot = 101 +cplot
    #     plt.subplot(cplot)

    plt.plot(t,signal)
    plt.title("Amplitud vs tiempo")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Amplitud [dB]")

def plotTransform(xft,ft):
    plt.title("Transformada ")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitud [dB]")
    plt.plot(xft,abs(ft))

##################################
# Espectrograma
##################################
def plotSpec(signal,fs_rate):
    plt.title("Espectrograma ")
    plt.specgram(signal,Fs=fs_rate)

####################################
# Mostrar todas las gráficas 
####################################
def getSignalTime(fs_rate,signal):
    
    signal_len = float(len(signal))
    tAux = float(signal_len)/float(fs_rate)
    t = np.linspace(0, tAux, signal_len)
    return t
def graphics(fs_rate,signal):
    #Se obtiene la grafica del audio vs tiempo
    plt.figure(1)
    plotSignalTime(signal,getSignalTime(fs_rate,signal))
    #Se obtiene la señal con el filtro paso bajo
    plt.figure(2)
    xFiltered,filteredSignal = lowFilter(fs_rate,1200,signal)
    plotSignalTime(filteredSignal,xFiltered)
    #Se obtiene la transformada de fourier
    xfft,fft = calcFFT(fs_rate,signal)
    #Se grafica la transformada del audio original
    plt.figure(3)
    plotTransform(xfft,fft)
    #Se grafica la transformada del audio con filtro paso bajo
    xlow,fftLow = calcFFT(fs_rate,filteredSignal)
    plt.figure(4)
    plotTransform(xlow,fftLow)

    #Se grafica el espectrograma del audio original
    plt.figure(5)
    plotSpec(signal,fs_rate)
    #Se grafica el espectrograma del audio con filtro paso bajo
    plt.figure(6)
    plotSpec(filteredSignal,fs_rate)
    plt.show()
    # plt.figure(1)
    # plt.subplot(311)
    # p1 = plt.plot(t, signal, "g") # plotting the signal
    # plt.xlabel('Time')
    # plt.ylabel('Amplitude')
    # plt.subplot(312)
    # p2 = plt.plot(freqs, FFT, "r") # plotting the complete fft spectrum
    # # plt.subplot(313)
    # # p3 = plt.specgram(signal,Fs=fs_rate)
    # # plt.xlabel('Time')
    # # plt.ylabel('Frequency')
    # # plt.figure(2)
    # plt.subplot(313)
    # p4 = plt.plot(xFilter,filteredSignal,"g")
    # plt.xlabel('dbs')
    # plt.ylabel('Frequency')
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


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

"""
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
    t = scipy.arange(0, secs, Ts) # time vector as scipy arange field / numpy.ndarra
    #Filtro de la señal
    xFilter,filteredSignal = lowFilter(fs_rate,1200,signal)
    xfft, FFT = calcFFT(fs_rate,signal)
    xFilter, FFT_filtered = calcFFT(xFilter,filteredSignal)
    freqs = scipy.fftpack.fftfreq(signal.size, t[1]-t[0])
    print("Filtered => \n")
    print(filteredSignal)
    #exit()
    if(flag=="1"):
        graphics(FFT, xfft, signal, t,fs_rate,FFT_filtered,xFilter)
    return
"""