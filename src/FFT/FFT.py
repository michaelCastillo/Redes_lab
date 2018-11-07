
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
 


def plotTransform(xft,ft,title):
    plt.title("Transformada "+title)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitud [dB]")
    plt.plot(xft,abs(ft))

def invTransform(ft,lenFreq):
    fourierTInv = scipy.ifft(ft)*lenFreq
    return fourierTInv
###################################
# Calculo de paso bajo
###################################

def lowFilter(fs,cutoff,signal):
    nq = fs/2
    lowcut = cutoff/nq
    numtaps = cutoff + 1
    filter_ = firwin(numtaps,cutoff = lowcut, window = 'hamming' )
    filteredSignal = lfilter(filter_,1.0,signal)
    len_signal = len(signal)#obtengo el largo de la señal
    len_signal = float(len_signal)
    time = float(len_signal)/float(fs)#genero el tiempo total del audio
    rate = np.arange(0,time,1.0/float(fs))
    return rate,filteredSignal

##################################
# Calculo de paso alto
##################################
def highFilter(fs,cutoff,signal):
    nq = fs/2
    highcut = cutoff/nq
    numtaps = cutoff + 1
    filter_ = firwin(numtaps,cutoff = highcut, window = 'hamming' ,pass_zero = False)
    filteredSignal = lfilter(filter_,1.0,signal)
    len_signal = len(signal)#obtengo el largo de la señal
    len_signal = float(len_signal)
    time = float(len_signal)/float(fs)#genero el tiempo total del audio
    rate = np.arange(0,time,1.0/float(fs))
    return rate,filteredSignal

##################################
# Calculo Paso banda
##################################
def passBandFilter(fs,backLimit,upLimit,signal):
    nq = fs/2
    lowcut = backLimit/nq
    highcut = upLimit/nq
    numtaps = upLimit + 1
    filter_ = firwin(numtaps,cutoff = [lowcut,highcut], window = 'hamming' ,pass_zero = False)
    filteredSignal = lfilter(filter_,1.0,signal)
    len_signal = len(signal)#obtengo el largo de la señal
    len_signal = float(len_signal)
    time = float(len_signal)/float(fs)#genero el tiempo total del audio
    rate = np.arange(0,time,1.0/float(fs))
    return rate,filteredSignal


##################################
# Grafica de señal vs tiempo
##################################

def plotSignalTime(signal,t,title):
    plt.plot(t,signal)
    plt.title("Amplitud vs tiempo "+title)
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Amplitud [dB]")
def getSignalTime(fs_rate,signal):
    
    signal_len = float(len(signal))
    tAux = float(signal_len)/float(fs_rate)
    t = np.linspace(0, tAux, signal_len)
    return t



##################################
# Espectrograma
##################################
def plotSpec(signal,fs_rate,title):
    plt.title("Espectrograma "+title)
    plt.specgram(signal,Fs=fs_rate)

####################################
# Mostrar todas las gráficas 
####################################

def graphics(fs_rate,signal):

    title_normal = "Audio original"
    #Graficas del audio original
    #Se obtiene la grafica del audio vs tiempo
    plt.figure(1)
    plt.subplot(311)
    plotSignalTime(signal,getSignalTime(fs_rate,signal),title_normal)
    #Se obtiene la transformada de fourier
    xfft,fft = calcFFT(fs_rate,signal)
    #Se grafica la transformada del audio original
    plt.subplot(312)
    plotTransform(xfft,fft,title_normal)

    #Se grafica el espectrograma del audio original
    plt.subplot(313)
    plotSpec(signal,fs_rate,title_normal)
    #Graficas con filtro paso bajo
    #Se obtiene la señal vs tiempo con el filtro paso bajo
    title_lowPass = "Filtro paso bajo"
    plt.figure(2)
    plt.subplot(311)
    xFiltered,filteredSignal = lowFilter(fs_rate,1200,signal)
    plotSignalTime(filteredSignal,xFiltered,title_lowPass)

    #Se grafica la transformada del audio con filtro paso bajo
    xlow,fftLow = calcFFT(fs_rate,filteredSignal)
    plt.subplot(312)
    plotTransform(xlow,fftLow,title_lowPass)

    
    #Se grafica el espectrograma del audio con filtro paso bajo
    plt.subplot(313)
    plotSpec(filteredSignal,fs_rate,title_lowPass)
    plt.show()

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