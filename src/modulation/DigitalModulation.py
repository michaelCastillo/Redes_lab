import numpy as np
import sys
from matplotlib import pyplot as plt
import scipy.io.wavfile as wavfile
sys.path.insert(0, '../SoundInterface')
import SoundOut as sin
sys.path.insert(0, '../Files InOut')
import fileDirector as FD
# import DigitalDemodulation as DDemodulation
import warnings
from scipy import signal as sg
warnings.filterwarnings("ignore")
from bitstring import Bits



def ASK(signal, fs, bitRate, title):
    t=np.arange(0, 1/bitRate, 1 / fs)
    A=2000
    B=50
    fc=fs/5
    
    carrier1 = A*np.cos(2*np.pi*fc*t)
    carrier2 = B*np.cos(2*np.pi*fc*t)
    y=[]
    for bit in signal:
        if (bit==1):
            y.extend(carrier1)
        else:
            y.extend(carrier2)
    
   #Portadoras
    # plt.figure(1)
    """
    plt.subplot(2,1,1)
    plt.title("ASK Carrier "+str(A)+" [db]")    
    plt.plot(carrier1)
    plt.subplot(2,1,2)    
    plt.title("ASK Carrier 2 "+str(B)+" [db]")    
    plt.plot(carrier2)
    plt.subplot(2,1,2)  
    plt.subplots_adjust(hspace = 1)
    #Señal
    """
    plt.figure(2)
    dCurve=genDigitalCurve(signal, fs, bitRate)
    plt.title("Señal Digital")
    plt.subplot(2,1,1)
    plt.plot(y)
    plt.title(title)    
    plt.subplot(2,1,2)
    plt.plot(dCurve)
    plt.subplots_adjust(hspace = 1)

    #Demodulacion
    demodulation(y,carrier1,carrier2,t,fs,bitRate)

    return np.array(y)
    
def genDigitalCurve(signal, fs, bitRate):
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

def getSignalTime(fs_rate, signal):
    signal_len = float(len(signal))
    tAux = float(signal_len) / float(fs_rate)
    t = np.linspace(0, tAux, signal_len)
    return t


def FSK(signal, fs, bitRate, title):
    A=10
    f1=fs/5 #Hz
    f2=fs/6 #Hz
    t=np.arange(0, 1/bitRate, 1 / fs)
    carrier1 = A*np.cos(2*np.pi*f1*t)
    carrier2 = A*np.cos(2*np.pi*f2*t)
    y=[]
    for bit in signal:
        if (bit==1):
            y.extend(carrier1)
        else:
            y.extend(carrier2)
    #Portadoras
    plt.figure(1)
    plt.subplot(2,1,1)
    plt.title("FSK Carrier "+str(f1)+" [Hz]")    
    plt.plot(carrier1)
    plt.subplot(2,1,2)    
    plt.title("FSK Carrier 2 "+str(f2)+" [Hz]")    
    plt.plot(carrier2)
    plt.subplot(2,1,2)  
    plt.subplots_adjust(hspace = 1)
    #Señal
    plt.figure(2)
    dCurve=genDigitalCurve(signal, fs, bitRate)
    plt.title("Señal Digital")
    plt.subplot(2,1,1)
    plt.plot(y)
    plt.title(title)    
    plt.subplot(2,1,2)
    plt.plot(dCurve)
    plt.subplots_adjust(hspace = 1)
    demodulation(y,carrier1,carrier2,t,fs,bitRate)

    return np.array(y)


def demodulation(signal,carrier_1,carrier_2,t,fs_rate,bitRate):
    signalTime = getSignalTime(fs_rate,signal)
    corr1 = sg.fftconvolve(signal,carrier_1,'same')
    corr2 = sg.fftconvolve(signal,carrier_2,'same')
    plt.figure(5)
    plt.subplot(3,1,1)
    plt.plot(t,carrier_1)
    plt.subplot(3,1,2)
    plt.plot(t,carrier_2)

    #Se obtienen las correlaciones
    corr1 = sg.medfilt(np.abs(corr1))
    corr2 = sg.medfilt(np.abs(corr2))
    #Se genera un array vacio para almacenar los bits obtenidos.
    arrayBits = []

    #Para recorrer el arreglo de correlaciones se debe mover fs_rate*tiempoBit para encontrar
    # cada maximo
    skip = fs_rate//bitRate  #muestras por 1 bit.
    bit_index = skip//2        # Indice del bit inicia en el centro de la primera señal.
    print(len(corr1))
    print(type(bit_index))
    depur = 0
    while(bit_index < len(corr1)):
        bitCorr1 = corr1[bit_index]
        bitCorr2 = corr2[bit_index]
        if( bitCorr1 > bitCorr2):
            arrayBits.append(1)
        else:
            arrayBits.append(0)
        bit_index = bit_index + skip
    print(str(arrayBits))
    result = depureMachine([0,1,0,0,0,1,]*10,arrayBits)
    if(result == 0):
        print("Demodulacion exitosa")
    else:
        print("Demodulacion fallida")
    plt.figure(6)
    plt.subplot(3,1,1)
    plt.plot(signalTime,corr1)
    plt.subplot(3,1,2)
    plt.plot(signalTime,corr2)
    plt.show()

def toBinary(x):
    x=Bits(int=x, length=32)
    return x.bin

def depureMachine(digitalSignal,digitalDemodulation):
    if(len(digitalSignal) != len(digitalDemodulation)):
        print("Las señales son diferentes! ")
    else:
        i = 0
        for bit in digitalSignal:
            if(bit != digitalDemodulation[i]):
                print("Las señales son diferentes! ")
                return 1
            i = i + 1
    print("Las señales son iguales! ")
    return 0


def mainDigitalModulation(modType,flag,fileName):
    flagTest=False
    test=[]
    binarySignal=[]
    if(flagTest):
        fs, signal = FD.openDigitalWav(fileName)
        #print("Signal:",signal)
        #toBinary = lambda x:Bits(int=x,length=32)
        signal=signal[0:10000]
        print("Original: ", signal)
        print(len(signal))
        binaryFunc = np.vectorize(toBinary)
        binarySignalAux=binaryFunc(signal)
        maxBinaryLenght=len("{0:b}".format((max(signal))))
        
        for index, value in enumerate(binarySignalAux):
            i=len(str(value))
            while i<maxBinaryLenght:
                binarySignal.append(0)
                i=i+1
            for bit in value:
                binarySignal.append(int(bit))
        binarySignal=np.array(binarySignal)
    else:
        test=[0,1,0,0,0,1]*10
        fs = 1000 #Frecuencia de muestreo en Hz
    
    bitRate=10 #Bit por segundo
    #ASK
    y=np.array([])
    
    if(modType=="ASK"):
        #Funcion que modula
        y=ASK(test, fs, bitRate, title="ASK "+fileName)

    #FSK
    if(modType=="FSK"):
        #Funcion que modula
        y=FSK(test, fs, bitRate, title="FSK "+fileName)
    #Escribir archivo .wav
    print(y)
    sin.writeWav(fileName+modType, fs, y)
    plt.show()
    
    #Plot de correlacionadores
    # demodulacion FSK
    



    # DDemodulation.mainDigitalDemodulation(flag, bitRate, fileName+modType)

    
    
mainDigitalModulation("ASK",1,"pruebita")