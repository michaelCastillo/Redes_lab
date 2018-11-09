
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
    # if(cplot%3 == 0):
    #     cplot = 101 +cplot
    #     plt.subplot(cplot)
    #     plt.figure(cplot)
    # else:
    #     cplot = 101 +cplot
    #     plt.subplot(cplot)

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

def graphics(fs_rate,signal,fileName, mpb):

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

    plt.subplots_adjust(hspace=0.9)
    plt.savefig('../plots/'+fileName+'1.png')
    mpb["value"] = 25
    #Graficas con filtro paso bajo
    #Se obtiene la señal vs tiempo con el filtro paso bajo
    title_lowPass = "filtro paso bajo"
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
    plt.subplots_adjust(hspace=0.9)
    mpb["value"] = 50
    plt.savefig('../plots/'+fileName+'2.png')
    ################## PASO ALTO ##################
    title_highPass = "filtro paso banda"
    plt.figure(4)

    plt.subplot(311)
    xFilteredHigh,filteredSignalHigh = passBandFilter(fs_rate,1200,7000,signal)
    plotSignalTime(filteredSignalHigh,xFilteredHigh,title_highPass)

    #Se grafica la transformada del audio con filtro paso bajo
    xhigh,fftHigh = calcFFT(fs_rate,filteredSignalHigh)
    plt.subplot(312)
    plotTransform(xhigh,fftHigh,title_highPass)

    
    #Se grafica el espectrograma del audio con filtro paso bajo
    plt.subplot(313)
    plotSpec(filteredSignalHigh,fs_rate,title_highPass)
    plt.subplots_adjust(hspace=0.9)
    
    plt.savefig('../plots/'+fileName+'4.png')
    mpb["value"] = 75

    ################## PASO BANDA ##################
    title_passBand = "filtro paso alto"
    plt.figure(3)
    plt.subplot(311)
    xFilteredPassBand,filteredPassBand = highFilter(fs_rate,7000,signal)
    plotSignalTime(filteredPassBand,xFilteredPassBand,title_passBand)

    #Se grafica la transformada del audio con filtro paso bajo
    xPassBand,fftPassBand = calcFFT(fs_rate,filteredPassBand)
    plt.subplot(312)
    plotTransform(xPassBand,fftPassBand,title_passBand)

    
    #Se grafica el espectrograma del audio con filtro paso bajo
    plt.subplot(313)
    plotSpec(filteredPassBand,fs_rate,title_passBand)
    plt.subplots_adjust(hspace=0.9)
    
    plt.savefig('../plots/'+fileName+'3.png')
    mpb["value"] = 100
    #plt.show()
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
    #plt.show()



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