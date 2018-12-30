import numpy as np
import sys
from matplotlib import pyplot as plt
import scipy.io.wavfile as wavfile
sys.path.insert(0, '../SoundInterface')
import SoundOut as sin
sys.path.insert(0, '../Files InOut')
import fileDirector as fileIn
from bitstring import Bits

import scipy
from scipy import integrate, fft
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
def genDigitalCurveDemodulation(signal, fs, bitRate):
    t=np.arange(0, 1/bitRate, 1 / fs)
    carrier1 = [1]*len(t)
    carrier2 = [0]*len(t)
    y=[]
    for bit in signal:
        if (bit==1):
            y.extend(carrier1)
        else:
            y.extend(carrier2)
    return np.array(y)
def twos_comp(val, bits):
    print((1 >> (bits - 1)))
    """compute the 2's complement of int value val"""
    if (val & (1 << (bits - 1))) != 0: # if sign bit is set e.g., 8bit: 128-255
        val = val - (1 << bits)        # compute negative value
    return val    

#dbCut: A que amplitud cortar (diferenciar 0 de 1)
#signal: Señal modulada en ASK
#flag: Bandera por si se quiere o no graficar
#bitRate: Bits leidos por segundo
#fs: Frecuencia muestreo
def askDemodulation(dbCut, signal, flag, bitRate, fs):
    aux=int(len(signal)/int(len(signal)/fs*bitRate))
    i=0
    dmSignal=[]
    while i<len(signal):
        bit=signal.item(i)
        if(signal.item(i)>dbCut):
            dmSignal.append(1)
        else:
            dmSignal.append(0)
        i=i+aux
    dmSignalAux=[]
    j=0
    bitAux=""
    for bit in dmSignal:
        if((j==32) and (bitAux!="")):
            dmSignalAux.append(Bits(bin=bitAux).int)
            j=0
            bitAux=""
        bitAux=bitAux+str(bit)
        j=j+1
    dmSignalAux=np.array(dmSignalAux)
    print("DmSignalAux: ",dmSignalAux)
    #plt.plot(dmSignalAux)
    sin.writeWav("pruebaDigital",fs,dmSignalAux)
    fs, signal=fileIn.openDigitalWav("pruebaDigital")
    x,y=calcFFT(fs, signal)
    #plotTransform(x,y,"prueba")
    plt.show()
    """
    plt.figure(3)
    dmSignalDigital=genDigitalCurveDemodulation(dmSignal, fs, bitRate)
    plt.title("Señal Demodulada ASK")
    plt.plot(dmSignalDigital)
    print("Señal demodulada ASK:")
    print(dmSignal)
    """    
#dbCut: A que amplitud cortar (diferenciar 0 de 1)
#signal: Señal modulada en ASK
#flag: Bandera por si se quiere o no graficar
#bitRate: Bits leidos por segundo
#fs: Frecuencia muestreo
def plotAskDemodulation(dbCut, signal, flag, bitRate, fs):
    dbCutVector=[dbCut]*len(signal) #Generación del vector
    plt.figure(4)
    plt.subplot(2,1,1)
    plt.title("Corte de la amplitud")
    plt.plot(signal)
    plt.plot(dbCutVector, color="k")

    plt.subplot(2,1,2)
    plt.title("Puntos donde se analizara la amplitud")
    #Creación del vector donde se analizara la amplitud
    #Para calcular la mitad de cada bit
    #Se divide el largo de la señal por la freq de muestreo,
    #y se multiplica por la cantidad de bits por segundo
    aux=len(signal)/fs*bitRate 
    #Se hace un vector que va desde la mitad de la señal dividido aux,
    #para que parta desde la mitad del primer bit, hasta 
    #el largo de la señal dividido aux.
    bitRateVector=range(int(int(len(signal)/aux)/2), len(signal), int(len(signal)/aux))
    
    plt.plot(signal)
    for xc in bitRateVector:
        plt.axvline(x=xc, color="k")

    plt.subplots_adjust(hspace = 1)

    if(flag==1):
        plt.show()
    


def mainDigitalDemodulation(flag, bitRate,fileName):
    fs, signal = fileIn.openDigitalWav(fileName)
    askDemodulation(15, signal, flag, bitRate, fs)
    plotAskDemodulation(15, signal, flag, bitRate, fs)
    #Demodulacion
    



mainDigitalDemodulation(1, 10, "pruebita"+"ASK")