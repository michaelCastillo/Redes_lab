from __future__ import print_function
#import scipy.io.wavfile as wavfile
import scipy
from scipy.signal import butter, lfilter, firwin
from scipy import integrate, fft
#Ifrom scipy import fftpack
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

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


def fmModulationFail(fs_rate, signal, freq, flag):
    timeSignalArray = getSignalTime(fs_rate,signal)
    time =  float(len(signal))/float(fs_rate)

    """
    time =  float(len(signal))/float(fs_rate)
    timeSignalArray = getSignalTime(fs_rate,signal)
    
    modFM , xFM = modulacion_FM(signal,len(signal),timeSignalArray,fs_rate,100,freq)

    xFFT, yFFT = calcFFT(fs_rate,modFM)
    plt.figure(4)
    plotTransform(xFFT,yFFT,"Señal modulada FM")
    plt.show()
    """
    """
    fm = fs_rate/2
    fc = 7*fm
    #Se establecen las frecuencias de muestreo
    time =  float(len(signal))/float(fs_rate)
    fsRateOriginal = fs_rate
    fsRateModulation = freq * 4
    #Se obtiene la señal original
    timeSignalArray = getSignalTime(fsRateOriginal,signal)
    #Se grafica la señal original
    plt.figure(1)
    plt.subplot(311)        
    #Se grafica la señal dec vs tiempo
    plotSignalTime(signal[:200],timeSignalArray[:200],"Señal original",False)    
    #Se obtiene la transformada de la señal
    xOriFFt,yOriFFt = calcFFT(fsRateOriginal,signal)
    plt.subplot(312)        
    plotTransform(xOriFFt,yOriFFt,"Señal original")


    #Si la frecuencia de muestreo obtenida desde la carrier es menor entonces, se deja
    # aquella que sea más alta con el fin de no interpolar hacia abajo la señal para que
    # no pierda presición. 
    interpolate = True
    if(fsRateModulation < fsRateOriginal):
        fsRateModulation = fsRateOriginal
        interpolate = False

    print("########### Portadora ############")
    print("fsRateOriginal => "+str(fsRateOriginal))
    print("fsRateModulation => "+str(fsRateModulation))
    print("Time => "+str(time))
    plt.figure(2)
    plt.subplot(3,1,1)
    #Se grafica la portadora en tiempo
    xCarrier,yCarrier = getCarrier(time,fsRateModulation,fc)
    plotSignalTime(yCarrier[:500],xCarrier[:500],"Señal portadora",False)
    print("############# LENS #############")
    print("Len xSignal => "+str(len(timeSignalArray)))
    print("Len xCarrier => "+str(len(xCarrier)))
    plt.subplot(3,1,2)
    #Se grafica la portadora en transformada
    xCarrier_t, yCarrier_t = calcFFT(fsRateModulation,yCarrier)
    plotTransform(xCarrier_t,yCarrier_t,"Transformada portadora")

    print(str(xCarrier))
    """
    ######### Modulacion ##########
    interp = interp1d(timeSignalArray, signal)
    timeWithNewRate = np.linspace(0, len(signal) / fs_rate,len(signal) * 10)
    dataInterpolated = interp(timeWithNewRate)
    lenDataInterpolated = len(timeWithNewRate)
    arrayTimeCarrier = np.linspace(0, len(signal) / fs_rate, lenDataInterpolated)
    newFrequency = fs_rate
    w = newFrequency*arrayTimeCarrier
    k=3 #solo al tanteo, puede ser cambiado; preguntarle al profe.
    integral = integrate.cumtrapz(dataInterpolated, arrayTimeCarrier, initial=0)
    modulatedSignal = np.cos(2*np.pi * w + k * integral)

    plt.figure(3)
    plt.subplot(3,1,1)
    plotSignalTime(signal[int(len(signal)/4):int(len(signal)/4) + 1000],timeSignalArray[int(len(signal)/4):int(len(signal)/4) + 1000],"Señal original",False)    
    plt.subplot(3,1,2)
    plotSignalTime(modulatedSignal[int(len(signal)/4):int(len(signal)/4) + 1000],timeWithNewRate[int(len(signal)/4):int(len(signal)/4) + 1000],"Señal modulata dec vs Tiempo",False)

    #Se obtiene la transformada de la modulacion FM
    xModFFT, yModFFT = calcFFT(newFrequency,modulatedSignal)
    plt.subplot(3,1,3)
    plotTransform(xModFFT,yModFFT,"Transformada Modulacion FM")
    """
    xToModulate = np.linspace(0,time,fc*time)
    yToModulate  = np.interp(xToModulate, timeSignalArray,signal)
    
    integral = integrate.cumtrapz(yToModulate, xToModulate, initial=0)
    w = fm*xToModulate
    KK=1
    product = np.cos(w*2*np.pi+KK*integral)

    #Se grafica la modulacion
    plt.figure(3)
    plt.subplot(3,1,1)
    plotSignalTime(product[:500],xToModulate[:500],"Señal modulata dec vs Tiempo",False)

    #Se obtiene la transformada de la modulacion FM
    xModFFT, yModFFT = calcFFT(fsRateModulation,product)
    plt.subplot(3,1,2)
    plotTransform(xModFFT,yModFFT,"Transformada Modulacion FM")
"""
    plt.show()
    

    
def modulacion_FM(señal,largo_señal,x,rate,K,freq):
    
    KK = K/100
    fm=rate/2 #se requiere la mitad de la frecuencia dada por la funcion "read" para obtener la frecuencia de muestreo
    fc=5*freq#se requiere una frecuencia portadora minimo 4 veces mayor que la frecuencia de muestreo
    wc=2*np.pi
    
    tiempo = float(largo_señal)/float(rate)#genero el tiempo del audio 
    #genero un vector de 0 hasta tiempo con intervalos del porte de la frecuencia
    x2 = np.linspace(0,tiempo,fc*tiempo)
    señal2=np.interp(x2, x,señal)
    
    
    A=1
    
    portadoraX=np.linspace(0,fc, fc*tiempo)
    
    señal_portadora = np.cos(wc*portadoraX)
    
    plt.figure(1)
    plt.subplot(311)
    xfft,yfft = calcFFT(rate,señal_portadora)
    plotTransform(xfft,yfft,"Portadora Transf")

    integral= integrate.cumtrapz(señal2,x2, initial=0)
    w= fm * x2
    señal_modulada=A*(np.cos(w*np.pi+KK*integral*np.pi))#por que se agrega un pi?
    
    plt.figure(2)
    plt.subplot(311)
    plt.title("Señal del Audio")
    plt.plot(x2,señal2,linewidth=0.4)
    plt.subplot(312)
    plt.title("Señal Portadora")
    plt.plot(portadoraX[:600],señal_portadora[:600],linewidth=0.4)
    plt.subplot(313)
    plt.title("Modulación FM "+str(K)+" %")
    plt.plot(x2,señal_modulada,linewidth=0.4)
    
    return señal_modulada,x2
    
def amModulation(fs_rate,signal, freq, flag):
    
    ##Esta es una modulacion con una función coseno de ejemplo
    #Se obtiene: 
    #    portadora : coseno con frecuencia = f Hz
    #   moduladora: coseno con frecuencia = f/2 Hz
    # con una frecuencia determinada. 
    
    #fs_rate = 4*freq #la frecuencia de muestreo debe ser al menos 4 la frecuencia (Por qué ? lo lei por ahi y funciona)
    time =  float(len(signal))/float(fs_rate)
    print("TIME => "+str(time))
    fs_rate_modulation = 4*freq  #Para que se cumpla el teorema de nyquist.

    newModulation(signal,fs_rate,fs_rate_modulation,time,freq)
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


def newModulation(signal,fsRateOriginal,fsRateModulation,time,fZero):
    #Eje x para la señal
    timeSignalArray = getSignalTime(fsRateOriginal,signal)
    ##
    #Señal original    
    ##
    plt.figure(1)
    plt.subplot(311)        
    plotSignalTime(signal,timeSignalArray,"Señal original",False)    
    xOriFFt,yOriFFt = calcFFT(fsRateOriginal,signal)
    plt.subplot(312)        
    plotTransform(xOriFFt,yOriFFt,"Señal original")

    ##
    #Señal portadora
    ##
    plt.figure(2)
    plt.subplot(311)    
    #xCarrier, yCarrier = getCarrierFunction(time,fsRateModulation,fZero)
    
    #Si la frecuencia de muestreo obtenida desde la carrier es menor entonces, se deja
    # aquella que sea más alta con el fin de no interpolar hacia abajo la señal para que
    # no pierda presición. 
    interpolate = True
    if(fsRateModulation < fsRateOriginal):
        fsRateModulation = fsRateOriginal
        interpolate = False
    
    xCarrier, yCarrier = getCarrier(time, fsRateModulation, fZero)
    plotSignalTime(yCarrier,xCarrier,"Señal portadora",False)    
    xCarryFFt,yCarryFFt = calcFFT(fsRateModulation,yCarrier)
    plt.subplot(312)        
    plotTransform(xCarryFFt,yCarryFFt,"Señal portadora")
    print("Len Signal =>  "+str(len(signal)))
    print("Len Carrier =>  "+str(len(yCarrier)))

    ########################
    #######  Modulación  ########
    ########################
    #Interpolación
    plt.figure(3)
    if(interpolate):
        signalInterpolate = np.interp(xCarrier,timeSignalArray,signal)
        print("Inretpolate "+str(len(signalInterpolate)))
        plt.subplot(311)
        plotSignalTime(signalInterpolate,xCarrier,"Señal original interpolada",False)    
        #Modulación
        modulateSignal = yCarrier*signalInterpolate
        xFFt,yFFt = calcFFT(fsRateModulation,modulateSignal)
        plt.subplot(312)    
        plotTransform(xFFt,yFFt,"Señal modulada")
    else:
        #Modulación
        modulateSignal = yCarrier*signal
        xFFt,yFFt = calcFFT(fsRateModulation,modulateSignal)
        plt.subplot(312)    
        plotTransform(xFFt,yFFt,"Señal modulada")


    ##########################
    #######  Demodulación  ########
    ##########################
    
    plt.figure(4)
    demodulated = modulateSignal*yCarrier
    demodulated = butter_lowpass_filter( demodulated,5000,fsRateModulation)
    xdemod,ydemod = calcFFT(fsRateModulation,demodulated)
    ydemod = ydemod*2
    plotSignalTime(xdemod,ydemod,"Señal demodulada",False)        
    plt.subplot(312)
    
    ##Señal inversa, para obtener la señal en el tiempo
    plt.subplot(311)
    inverse = invTransform(ydemod,len(xdemod))
    plotSignalTime(inverse,xCarrier,"Señal original",False)    

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

def getCarrierSin(time, fs_rate, fZero):
    # hz = c/s   .=>
    print("time => " + str(time))
    print("fs_rate => " + str(fs_rate))
    print("Time fs_rate +. " + str(time * fs_rate))
    x = np.arange(0, time, 1 / fs_rate)
    # x = np.linspace(0, time, num=time*fs_rate, endpoint=True)
    y = np.sin(fZero * 2 * np.pi * x)
    print("Y +. " + str(len(y)))
    # f = interp1d(x, y)
    # print("F => "+str(len(f) ))
    return x, y

def fmModulation(signal,x,fs_rate,fZero,time):
    acumIntegral = 100*integrate.cumtrapz(signal, x, initial=0)
    x = np.arange(0, time, 1 / fs_rate)
    # x = np.linspace(0, time, num=time*fs_rate, endpoint=True)
    y = np.cos(fZero * 2 * np.pi * x + acumIntegral)
    return y
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

# SE REALIZA UN FILTRO PASO BAJO A LA DEMODULACION
def demodulation(carrier, demod, fs_rate, plotType):
    #Filtro paso bajo
    y = butter_lowpass_filter(demod[1], 2000, fs_rate, order=5)
    demodToPlot = demod[0], y
    plotCarrierAndModulizerFunctions(carrier, demodToPlot, fs_rate, plotType)

def getSampleFunction(time, samplesPerSec, fZero):
    t = np.linspace(0, 0.1, 2 * samplesPerSec, endpoint=False)
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
