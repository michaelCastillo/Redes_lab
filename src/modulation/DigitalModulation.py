import math
import sys
import wave

import numpy as np
import scipy.io.wavfile as wavfile
from matplotlib import pyplot as plt
from numpy.ma.core import concatenate


sys.path.insert(0, '../SoundInterface')
import SoundOut as sin
sys.path.insert(0, '../Files InOut')
import fileDirector as FD
# import DigitalDemodulation as DDemodulation
import warnings
from scipy import signal as sg
warnings.filterwarnings("ignore")
from bitstring import Bits


def depureMachine(digitalSignal,digitalDemodulation):
    errors = 0
    i = 0
    for bit in digitalSignal:
        if(bit != digitalDemodulation[i]):
            errors = errors + 1
        i = i + 1
    if(errors != 0):
        return float(errors)*100/float(len(digitalSignal))
    return 0



def ASK(signal, fs, bitRate, title):
    t=np.arange(0, 1/bitRate, 1 / fs)
    A=2000
    B=50
    fc=fs/5
    
    carrier1 = A*np.cos(2*np.pi*fc*t)
    carrier2 = B*np.cos(2*np.pi*fc*t)
    y=[]
    for sample in signal:
        sample = "{0:08b}".format(sample)
        for bit in sample:
            if(bit == "1"):
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
    #Se agrega ruido gausiano a la señal modulada
    mean = 0
    std = 1
    noise = np.random.normal(0.0, 2000, len(y))
    #y = y + noise
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
    
    ask_demodulation(y,carrier1,carrier2,t,fs,bitRate)

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
    f1=fs/4+2000
    f2=fs/4-2000
    t=np.arange(0, 1/bitRate, 1 / fs)
    
    carrier1 = A*np.cos(2*np.pi*f1*t)
    carrier2 = A*np.cos(2*np.pi*f2*t)
    y=[]
    for sample in signal:
        sample = "{0:08b}".format(sample)
        for bit in sample:
            if(bit == "1"):
                y.extend(carrier1)
            else:
                y.extend(carrier2)

    print("tiempo de bit: "+str(1/bitRate))
    print("numero de muestras totales: "+str(i*len(t)))
    print("Segundos totales!: "+str(i*len(t)/fs))
    #Portadoras

    #Señal
    noise = np.random.normal(0.0, 5, len(y))
    #y = y + noise
    dCurve=genDigitalCurve(signal, fs, bitRate)
    
    
    return np.array(signal)

def ask_demodulation(signal,carrier_1,carrier_2,t,fs_rate,bitRate):
    signalTime = getSignalTime(fs_rate,signal)
    corr1 = sg.fftconvolve(signal,carrier_1,'same')
    #corr2 = sg.fftconvolve(signal,carrier_2,'same')
    # Se obtienen las correlaciones
    corr1 = sg.medfilt(np.abs(corr1))
    #corr2 = sg.medfilt(np.abs(corr2))
    # Se genera un array vacio para almacenar los bits obtenidos.
    arrayBits = []
    # Para recorrer el arreglo de correlaciones se debe mover fs_rate*tiempoBit para encontrar
    # cada maximo
    skip = fs_rate//bitRate  # muestras por 1 bit.
    bit_index = skip//2   # Indice del bit inicia en el centro de la primera señal.
    print(len(corr1))
    while(bit_index < len(corr1)):
        bitCorr1 = corr1[bit_index]
        bitCorr2 = max(corr1) - corr1[bit_index] 
        if( bitCorr1 > bitCorr2):
            arrayBits.append(1)
        else:
            arrayBits.append(0)
        bit_index = bit_index + skip

    dCurve=genDigitalCurve(arrayBits, fs_rate, bitRate)
    plt.figure(5)
    plt.subplot(2,1,1)
    plt.plot(arrayBits)
    plt.title("arrayBits")
    plt.subplot(2,1,2)
    plt.plot(dCurve)
    plt.title("Demodulada")
    plt.subplots_adjust(hspace = 1)

    print(str(arrayBits))

    result = depureMachine([0,1,0,0,0,1,]*10,arrayBits)

    if(result == 0):
        print("Demodulacion exitosa")
    else:
        print("Demodulacion fallida")

    plt.figure(6)
    plt.subplot(3,1,1)
    plt.plot(signalTime,corr1)
    plt.title("Correlacion 1")
    plt.subplot(3,1,2)
    #plt.plot(signalTime,corr2)
    #plt.title("Correlacion 2")
    plt.subplots_adjust(hspace = 1)
    plt.show()


def fsk_demodulation(signal,carrier_1,carrier_2,t,fs_rate,bitRate):
    signalTime = getSignalTime(fs_rate,signal)
    corr1 = np.correlate(signal,carrier_1,'same')
    corr2 = np.correlate(signal,carrier_2,'same')
    plt.figure(5)
    plt.subplot(3,1,1)
    plt.plot(t,carrier_1)
    plt.subplot(3,1,2)
    plt.plot(t,carrier_2)

    #Se obtienen las correlaciones
    # corr1 = sg.medfilt(np.abs(corr1))
    # corr2 = sg.medfilt(np.abs(corr2))
    #Hilbert
    corr1_raw = corr1
    corr2_raw = corr2
    corr1 = np.abs(sg.hilbert(corr1))
    corr2 = np.abs(sg.hilbert(corr2))
    

    #Se genera un array vacio para almacenar los bits obtenidos.
    arrayBits = []

    #Para recorrer el arreglo de correlaciones se debe mover fs_rate*tiempoBit para encontrar
    # cada maximo
    skip = fs_rate//bitRate  #muestras por 1 bit.
    print("numero de bits: "+str(len(corr1)/skip))
    bit_index = skip//2        # Indice del bit inicia en el centro de la primera señal.
    print(len(corr1))
    print(type(bit_index))
    depur = 0
    indexDemod1 = []
    indexDemod2 = []
    while(bit_index < len(corr1)):
        bitCorr1 = corr1[bit_index]
        bitCorr2 = corr2[bit_index]
        if( bitCorr1 > bitCorr2):
            arrayBits.append(1)
            indexDemod1.append(bit_index)
        else:
            indexDemod2.append(bit_index)
            arrayBits.append(0)
        bit_index = bit_index + skip
    print(str(arrayBits))
    

    valuesToPointCorr1 = [0]*int(len(corr1))
    valuesToPointCorr2 = [0]*int(len(corr2))
    for i in indexDemod1:
        valuesToPointCorr1[i] = corr1[i]
    i = 0
    for i in indexDemod2:
        valuesToPointCorr2[i] = corr2[i]





    # plt.figure(6)
    plt.figure(6)
    plt.subplot(3,1,1)
    plt.title("Hilbert portadora 1")
    # plt.plot(signalTime,np.abs(corr1_raw))
    plt.plot(signalTime,np.abs(valuesToPointCorr1),"*-")
    plt.plot(signalTime,np.abs(valuesToPointCorr2),"*-",color="red")
    plt.plot(signalTime,corr1,color="green")
    plt.subplot(3,1,2)
    plt.title("Hilbert portadora 2")
    plt.plot(signalTime,np.abs(valuesToPointCorr2),"*-")
    plt.plot(signalTime,np.abs(valuesToPointCorr1),"*-",color="red")
    plt.plot(signalTime,corr2,color="green")
    plt.show()
    return arrayBits

def toBinary(x):
    x=Bits(int=x, length=32)
    return x.bin




def mainDigitalModulation(modType,flag,fileName):
    flagTest=True
    test=[]
    binarySignal=[]
    trenBinaries = []
    if(flagTest):
        fs, signal = FD.openDigitalWav(fileName)
        waveData = wave.open("../../Wav/"+fileName+".wav","rb")
        byteSignal = waveData.readframes(waveData.getnframes())
        for i in byteSignal:
            binaryValue = '{0:08b}'.format(i)
            binList = list(binaryValue)
            binList = list(map(int, binList))
            np.concatenate((binarySignal,np.array(binList)),axis=0)
        print(len(binarySignal)) ##HAy que concatenar los arrays
        
        

    else:
        test=[0,1,0,0,0,1]*1000
        print(test)
        plt.figure(4)
        plt.subplot(3,1,1)
        plt.plot(test)
        plt.title("test")
        #plt.subplot(3,1,2)
        #plt.plot(t,carrier_2)
        #plt.title("Carrier 2")
        #plt.subplots_adjust(hspace = 1)
        fs = 40000 #Frecuencia de muestreo en Hz
    
    bitRate=1000 #Bit por segundo

    """
    y=np.array([])    
    if(modType=="ASK"):
        #Funcion que modula
        y=ASK(test, fs, bitRate, title="ASK "+fileName)

    #FSK
    if(modType=="FSK"):
        #Funcion que modula
        print("Binary Signal => "+str(binarySignal))
        y=FSK(test, fs, bitRate, title="FSK "+fileName)
    #Escribir archivo .wav
    
    sin.writeWav(fileName+modType, fs, y)
    plt.show()
    #Plot de correlacionadores
    #demodulacion FSK
    #DDemodulation.mainDigitalDemodulation(flag, bitRate, fileName+modType)
    """
mainDigitalModulation("FSK",1,"pasoDigital")