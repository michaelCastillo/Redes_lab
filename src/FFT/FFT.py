from __future__ import print_function
#import scipy.io.wavfile as wavfile
import scipy
from scipy.signal import butter, lfilter, firwin
#Ifrom scipy import fftpack
import numpy as np
from matplotlib import pyplot as plt
#from scipy.interpolate import interp1d

#CALCULO DE LA TRANSFORMADA DE FOURIER
def calcFFT(fs_rate, signal):
    #Se calcula la transformada
    fft = scipy.fft(signal)
    #Normalizar transformada dividiendola por el largo de la señal
    fftNormalized = fft / len(signal)
    #Genera las frecuencias de muestreo de acuerdo al largo de fftNormalize y el inverso de la tasa de muetreo
    xfft = np.fft.fftfreq(len(fftNormalized), 1 / fs_rate)
    return xfft, fftNormalized

#GRÁFICA DE LA TRANSFORMADA DE FOURIER
def plotTransform(xft, ft, title):
    #Titulo gráfico
    plt.title("Transformada " + title)
    #Ejes x e y
    plt.xlabel("Frecuencia [Hz]")
    plt.ylabel("Amplitud [dB]")
    plt.plot(xft, abs(ft))

#CÁLCULO DE LA TRANSFORMADA INVERSA FOURIER
def invTransform(ft, lenFreq):
    #Se calcula la transformada inversa
    fourierTInv = scipy.ifft(ft) * lenFreq
    return fourierTInv

#CALCULO DE FILTRO PASO BAJO
def lowFilter(fs, cutoff, signal):
    nq = fs / 2
    lowcut = cutoff / nq
    numtaps = cutoff + 1
    #Generar el filtro paso bajo, mediante ventana hamming que usa el coseno
    filter_ = firwin(numtaps, cutoff=lowcut, window='hamming')
    #Se aplica el filtro a la señal
    filteredSignal = lfilter(filter_, 1.0, signal)
    #Se obtiene el largo de la señal
    len_signal = len(signal)
    len_signal = float(len_signal)
    #Se genera el tiempo total del audio
    time = float(len_signal) / float(fs)
    rate = np.arange(0, time, 1.0 / float(fs))
    return rate, filteredSignal

#CALCULO DE FILTRO PASO ALTO
def highFilter(fs, cutoff, signal):
    nq = fs / 2
    highcut = cutoff / nq
    numtaps = cutoff + 1
    #Generar el filtro paso ALTO, mediante ventana hamming que usa el coseno
    filter_ = firwin(numtaps, cutoff=highcut, window='hamming', pass_zero=False)
    #Se aplica el filtro a la señal
    filteredSignal = lfilter(filter_, 1.0, signal)
    #Se obtiene el largo de la señal
    len_signal = len(signal)
    len_signal = float(len_signal)
    #Se genera el tiempo total del audio
    time = float(len_signal) / float(fs)
    rate = np.arange(0, time, 1.0 / float(fs))
    return rate, filteredSignal

#CALCULO DE FILTRO PASO BANDA
def passBandFilter(fs, backLimit, upLimit, signal):
    nq = fs / 2
    lowcut = backLimit / nq
    highcut = upLimit / nq
    numtaps = upLimit + 1
    #Generar el filtro paso ALTO, mediante ventana hamming que usa el coseno
    filter_ = firwin(numtaps, cutoff=[lowcut, highcut], window='hamming', pass_zero=False)
    #Se aplica el filtro a la señal
    filteredSignal = lfilter(filter_, 1.0, signal)
    #Se obtiene el largo de la señal
    len_signal = len(signal)
    len_signal = float(len_signal)
    #Se genera el tiempo total del audio
    time = float(len_signal) / float(fs)
    rate = np.arange(0, time, 1.0 / float(fs))
    return rate, filteredSignal

#GRAFICA DE SEÑAL VS TIEMPO
def plotSignalTime(signal, t, title, dot):
    # if(cplot%3 == 0):
    #     cplot = 101 +cplot
    #     plt.subplot(cplot)
    #     plt.figure(cplot)
    # else:
    #     cplot = 101 +cplot
    #     plt.subplot(cplot)

    if (dot):
        plt.plot(t, signal, '*-')
    else:
        plt.plot(t, signal)
    #Titulo del gráfico
    plt.title("Amplitud vs tiempo " + title)
    #Ejes x e y
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Amplitud [dB]")
    plt.subplots_adjust(hspace=1)

def getSignalTime(fs_rate, signal):
    signal_len = float(len(signal))
    tAux = float(signal_len) / float(fs_rate)
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
def fmModulation(fs_rate, signal, freq, flag):
    #Eje x para la señal
    timeSignalArray = getSignalTime(8192,signal)
    ##
    #Señal original    
    ##
    #plt.figure(1)
    #plt.subplot(311)        
    #plotSignalTime(signal,timeSignalArray,"Señal original",True)    
    #xOriFFt,yOriFFt = calcFFT(freq,signal)
    #plt.subplot(312)        
    #plotTransform(xOriFFt,yOriFFt,"Señal original")
    
    fs_rate=freq*4
    if(flag==1):
        timeSignalArray = np.arange(44100.0) / 44100.0
        signal = np.cos(freq * np.pi * timeSignalArray)
    time =  float(len(signal))/float(fs_rate)
    xCarrier, yCarrier = getCarrier(time, fs_rate, freq)

    product = np.zeros_like(signal, dtype=float)
    signalIntegrate=np.cumsum(signal)
    for i, t in enumerate(timeSignalArray):
        product[i] = np.cos( np.pi * (fs_rate * t + signal[i]))
    plt.subplot(3, 1, 1)

    #Original
    plt.figure(1)
    plt.subplot(311)        
    plotSignalTime(signal,timeSignalArray,"Señal original",False)    
    xOriFFt,yOriFFt = calcFFT(freq,signal)
    plt.subplot(312)
    
    plotTransform(xOriFFt,yOriFFt,"Señal original")
    
    #Portadora
    plt.figure(2)
    plt.subplot(311)    
    plt.ylabel('Amplitud')
    plt.xlabel('Señal Portadora')
    plotSignalTime(yCarrier,xCarrier,"Señal portadora",False)    
    xCarryFFt,yCarryFFt = calcFFT(fs_rate,yCarrier)

    plt.subplot(312)        
    plotTransform(xCarryFFt,yCarryFFt,"Señal portadora")
    

    #Modulacion
    plt.figure(3)
    plt.subplot(311)    
    plt.ylabel('Amplitud')
    plt.xlabel('Señal Modulada')
    plt.plot(product)
   
    xCarryFFt,yCarryFFt = calcFFT(fs_rate,product)
    plt.subplot(312)        
    plotTransform(xCarryFFt,yCarryFFt,"Señal modulada")
    
    plt.show()


def amModulation(fs_rate,signal, freq, flag):
    
    ##Esta es una modulacion con una función coseno de ejemplo
    #Se obtiene: 
    #    portadora : coseno con frecuencia = f Hz
    #   moduladora: coseno con frecuencia = f/2 Hz
    # con una frecuencia determinada. 
    
    #fs_rate = 4*freq #la frecuencia de muestreo debe ser al menos 4 la frecuencia (Por qué ? lo lei por ahi y funciona)
    time =  float(len(signal))/float(fs_rate)
    print("TIME => "+str(time))
    fs_rate = 4*freq  #Para que se cumpla el teorema de nyquist.

    newModulation(signal,fs_rate,time,freq)
    plt.show()


#def getFrequencyTimePlot(samplingFrequency, signalData):
    # Plot the signal read from wav file
#    plt.subplot(211)
#    plt.title('Spectrogram of a wav file with piano music')
#    plt.plot(signalData)
#    plt.xlabel('Sample')
#    plt.ylabel('Amplitude')
#    plt.subplot(212)
#    plt.specgram(signalData, Fs=samplingFrequency)
#    plt.xlabel('Time')
#    plt.ylabel('Frequency')
#    plt.show()

# CREA EL FILTRO PASO BAJO
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

# USAR EL FILTRO PASO BAJO SOBRE LA SEÑAL
def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    #Aplicar filtro a la señal
    y = lfilter(b, a, data)
    return y


def newModulation(signal,fs_rate,time,fZero):
    #Eje x para la señal
    timeSignalArray = getSignalTime(8192,signal)
    ##
    #Señal original    
    ##
    plt.figure(1)
    plt.subplot(311)        
    plotSignalTime(signal,timeSignalArray,"Señal original",False)    
    xOriFFt,yOriFFt = calcFFT(fZero,signal)
    plt.subplot(312)        
    plotTransform(xOriFFt,yOriFFt,"Señal original")

    ##
    #Señal portadora
    ##
    plt.figure(2)
    plt.subplot(311)    
    #xCarrier, yCarrier = getCarrierFunction(time,fs_rate,fZero)
    xCarrier, yCarrier = getCarrier(time, fs_rate, fZero)
    plotSignalTime(yCarrier,xCarrier,"Señal portadora",False)    
    xCarryFFt,yCarryFFt = calcFFT(fs_rate,yCarrier)
    plt.subplot(312)        
    plotTransform(xCarryFFt,yCarryFFt,"Señal portadora")
    
    ##
    #Modulacion
    ##
    plt.figure(3)
    signalInterpolate = np.interp(xCarrier,timeSignalArray,signal)
    print("Inretpolate "+str(len(signalInterpolate)))
    plt.subplot(311)
    plotSignalTime(signalInterpolate,xCarrier,"Señal original interpolada",True)    
    modulateSignal = yCarrier*signalInterpolate
    xFFt,yFFt = calcFFT(fs_rate,modulateSignal)
    plt.subplot(312)    
    plotTransform(xFFt,yFFt,"Señal modulada")
    ##
    #Demodulacion
    ##
    plt.figure(4)
    demodulated = modulateSignal*yCarrier
    xdemod,ydemod = calcFFT(fZero,demodulated)
    plotSignalTime(xdemod,ydemod,"Señal demodulada",False)        
    plt.subplot(312)
    print("x0 => "+str(xdemod[0]))
    print("xF => "+str(xdemod[len(xdemod)-1]))
    plt.subplot(312)
    plotTransform(xdemod,ydemod,"Señal demodulada")
    print("len signal => "+str(len(signal)))

def modulation(signal,fs_rate,fZero):
    
    #Se tienen que tener la misma cantidad de datos para poder multiplicar  
    #Hacerlo con un coseno pequeño 
    # en el tiempo hay que ver
    # : señal que esta entrando
    # señal portadora
    # multiplicación de los 2
    # y los mismos 3 en frecuencia.
    plt.figure(1)
    plt.subplot(311)
    time = 0.01
    # xCarrier, yCarrier = getCarrierFunction(time,fs_rate,fZero)
    # Se obtiene una función portadora
    xCarrier, yCarrier = getCarrier(time, len(signal), fZero)
    plotSignalTime(yCarrier, xCarrier, "Amplitud vs t PORTADORA", True)
    # Se calcula la transformada de la portadora y se puede notar que
    # corresponde a dos impulsos, o al menos a eso se acerca a medida
    # de que le damos más muestras por segundo.
    plt.subplot(312)
    xfft, fft = calcFFT(len(signal) / 10, yCarrier)
    plotTransform(xfft, fft, "Transformada señal portadora")

    # Se obtiene una funcion de ejemplo para modular
    # xMod,yMod = getSampleFunction(time,fs_rate,fZero)
    # Se usa el audio obtenido
    # f = getCarrierWithAudio(time, fs_rate, len(signal), fZero)
    xMod = getSignalTime(fs_rate, signal)
    yMod = signal

    carrier = [xCarrier, yCarrier]
    modulizer = [xMod, yMod]
    plotCarrierAndModulizerFunctions(carrier, modulizer, fs_rate, 2)

    # Demodulación
    #
    yDemod = yMod * yCarrier
    demod = xMod, yDemod
    demodulation(carrier, demod, fs_rate, 2)

# OBTENER UNA PORTADORA ADECUADA
def getCarrier(time, fs_rate, fZero):
    # hz = c/s   .=>
    print("time => " + str(time))
    print("fs_rate => " + str(fs_rate))
    print("Time fs_rate +. " + str(time * fs_rate))
    x = np.arange(0, time, 1 / fs_rate)
    # x = np.linspace(0, time, num=time*fs_rate, endpoint=True)
    y = np.cos(fZero * 2 * np.pi * x)
    print("Y +. " + str(len(y)))
    # f = interp1d(x, y)
    # print("F => "+str(len(f) ))
    return x, y

# SE REALIZA UN FILTRO PASO BAJO A LA DEMODULACION
def demodulation(carrier, demod, fs_rate, plotType):
    #Filtro paso bajo
    y = butter_lowpass_filter(demod[1], 2000, fs_rate, order=5)
    demodToPlot = demod[0], y
    plotCarrierAndModulizerFunctions(carrier, demodToPlot, fs_rate, plotType)


#def getCarrierExample(time, samplesPerSec, fZero):
    # time = np.arange(0,time,1.0/samplesPerSec)
#    t = np.linspace(0, 2, 2 * samplesPerSec, endpoint=False)
#    x = np.sin(fZero * 2 * np.pi * t)
    # time = np.linspace(0,time,samplesPerSec*4*fZero,endpoint=False)
#    return t, x


def getSampleFunction(time, samplesPerSec, fZero):
    t = np.linspace(0, 2, 2 * samplesPerSec, endpoint=False)
    fZero = fZero / 2
    x = np.sin(fZero * 2 * np.pi * t)
    return t, x

#GRAFICAR DE TAL MANERA QUE APAREZCAN TODOS.
def plotCarrierAndModulizerFunctions(carrier, modulizer, fs_rate, nFigure):
    plt.figure(nFigure)
    plt.subplot(311)
    #Grafico de funcion portadora vs amplitud
    plotSignalTime(carrier[1], carrier[0], "Amplitud vs t PORTADORA", True)
    plt.subplot(312)
    #Grafico de funcion modularizadora bs amplitud
    plotSignalTime(modulizer[1], modulizer[0], "Amplitud vs t MODULARIZADORA", True)

    #print("len carr = ", len(carrier[0]))
    #print("len mod= ", len(modulizer[0]))

    # Señal modulada
    modulizedSignal = carrier[1] * modulizer[1]
    # Transformada de Fourier a modulada
    xfft, fftMod = calcFFT(fs_rate, modulizedSignal)
    # Transformada de Fourier a original
    xModulizer, yModulizer = calcFFT(fs_rate, modulizer[1])
    # Transformada de Fourier a portadora
    xPor, yPor = calcFFT(fs_rate, carrier[1])
    plt.figure(3)
    #Grafico señal portadora
    plt.subplot(311)
    plotTransform(xPor, yPor, "Señal Portadora")
    #Grafico señal original
    plt.subplot(312)
    plotTransform(xModulizer, yModulizer, "Señal Original")
    #Grafico de señal modulada
    plt.subplot(313)
    plotTransform(xfft, fftMod, "Señal Modulada")