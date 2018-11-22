
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

def plotSignalTime(signal,t,title,dot):
    # if(cplot%3 == 0):
    #     cplot = 101 +cplot
    #     plt.subplot(cplot)
    #     plt.figure(cplot)
    # else:
    #     cplot = 101 +cplot
    #     plt.subplot(cplot)

    if(dot):
        plt. plot(t,signal,'*-') 
    else:
        plt. plot(t,signal) 
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
    
    modulation(signal,fs_rate,10000)
    plt.show()
"""
    title_normal = "Audio original"
    #Graficas del audio original
    #Se obtiene la grafica del audio vs tiempo
    plt.figure(1)
    plt.subplot(311)
    plotSignalTime(signal,getSignalTime(fs_rate,signal),title_normal,False)
    

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
    title_lowPass = "filtro paso bajo"
    plt.figure(2)
    plt.subplot(311)
    xFiltered,filteredSignal = lowFilter(fs_rate,1200,signal)
    plotSignalTime(filteredSignal,xFiltered,title_lowPass,False)

    #Se grafica la transformada del audio con filtro paso bajo
    xlow,fftLow = calcFFT(fs_rate,filteredSignal)
    plt.subplot(312)
    plotTransform(xlow,fftLow,title_lowPass)

    
    #Se grafica el espectrograma del audio con filtro paso bajo
    plt.subplot(313)
    plotSpec(filteredSignal,fs_rate,title_lowPass)

    ################## PASO ALTO ##################
    title_highPass = "filtro paso banda"
    plt.figure(4)
    plt.subplot(311)
    xFilteredHigh,filteredSignalHigh = passBandFilter(fs_rate,1200,7000,signal)
    plotSignalTime(filteredSignalHigh,xFilteredHigh,title_highPass,False)

    #Se grafica la transformada del audio con filtro paso bajo
    xhigh,fftHigh = calcFFT(fs_rate,filteredSignalHigh)
    plt.subplot(312)
    plotTransform(xhigh,fftHigh,title_highPass)

    
    #Se grafica el espectrograma del audio con filtro paso bajo
    plt.subplot(313)
    plotSpec(filteredSignalHigh,fs_rate,title_highPass)
    
    ################## PASO BANDA ##################
    title_passBand = "filtro paso alto"
    plt.figure(3)
    plt.subplot(311)
    xFilteredPassBand,filteredPassBand = highFilter(fs_rate,7000,signal)
    plotSignalTime(filteredPassBand,xFilteredPassBand,title_passBand,False)

    #Se grafica la transformada del audio con filtro paso bajo
    xPassBand,fftPassBand = calcFFT(fs_rate,filteredPassBand)
    plt.subplot(312)
    plotTransform(xPassBand,fftPassBand,title_passBand)

    
    #Se grafica el espectrograma del audio con filtro paso bajo
    plt.subplot(313)
    plotSpec(filteredPassBand,fs_rate,title_passBand)

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
    #plt.show()
 """


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

def modulation(signal,fs_rate,fZero):
    
    #Se tienen que tener la misma cantidad de datos para poder multiplicar  
    #Hacerlo con un coseno pequeño 
    # en el tiempo hay que ver
    # : señal que esta entrando
    # señal portadora
    # multiplicación de los 2 
    # y los mismos 3 en frecuencia.

    #Se obtiene una función portadora. 
    plt.figure(1)
    plt.subplot(311)
    xCarrier, yCarrier = getCarrierFunction(0.01,0.01,fZero)
    plotSignalTime(yCarrier,xCarrier,"Amplitud vs t PORTADORA",True)

    #Se calcula la transformada de la portadora y se puede notar que
    # corresponde a dos impulsos, o al menos a eso se acerca a medida
    # de que le damos más muestras por segundo.
    plt.subplot(312)
    xfft,fft = calcFFT(fs_rate,yCarrier)
    plotTransform(xfft,fft,"Transformada señal portadora")

    #Se obtiene una funcion de ejemplo para modular
    xMod,yMod = getSampleFunction(0.005,0.01,fZero)

    carrier = [xCarrier,yCarrier]
    modulizer = [xMod,yMod]
    plotCarrierAndModulizerFunctions(carrier,modulizer,2)
    
    #Se genera la modulación entre los dos cosenos.


def getCarrierFunction(time, samplesPerSec,fZero):
    time = np.linspace(0,time,samplesPerSec*4*fZero,endpoint=False)
    amplitude = np.cos(time*2*np.pi*fZero)
    return time,amplitude

def getSampleFunction(time,samplesPerSec,fZero):
    time = np.linspace(0,time,samplesPerSec*4*fZero,endpoint=False)
    amplitude = np.cos(time*2*np.pi*(fZero/2))
    return time,amplitude

def plotCarrierAndModulizerFunctions(carrier,modulizer,nFigure):
    plt.figure(nFigure)
    plt.subplot(311)
    plotSignalTime(carrier[1],carrier[0],"Amplitud vs t PORTADORA",True)
    plt.subplot(312)
    plotSignalTime(modulizer[1],modulizer[0],"Amplitud vs t MODULARIZADORA",True)

    print("len carr = ",len(carrier[0]))
    print("len mod= ",len(modulizer[0]))
    modulizedSignal =  carrier[1]*modulizer[1]
    xfft,fftMod = calcFFT(4*10000,modulizedSignal)
    xModulizer,yModulizer = calcFFT(4*10000,modulizer[1])
    xPor,yPor = calcFFT(4*10000,carrier[1])
    plt.figure(3)

    plt.subplot(311)
    plotTransform(xPor,yPor, "Señal Portadora")

    plt.subplot(312)
    plotTransform(xModulizer,yModulizer, "Señal Original")

    plt.subplot(313)
    plotTransform(xfft,fftMod, "Señal Modulada")
    
