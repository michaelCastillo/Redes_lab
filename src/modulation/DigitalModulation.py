import numpy as np
import sys
import math
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
import wave as w
import threading


def ASK(signal, fs, bitRate, threads, title):
    t=np.arange(0, 1/bitRate, 1 / fs)
    A=2000
    B=50
    fc=fs/5
    
    carrier1 = A*np.cos(2*np.pi*fc*t)
    carrier2 = B*np.cos(2*np.pi*fc*t)
    samplesPerThreads = len(signal)//threads
    start = 0
    end = samplesPerThreads
    i = 0
    # Lista de hebras
    threadList = []
    while i<threads:
        # Se inicializa la hebra
        mythread = threading.Thread(target=signalOperator, args=(start, end, signal, t, fs, i, title, carrier1, carrier2)) 
        # Se agrega a la lista
        threadList.append(mythread)
        # Se inicia
        mythread.start()
        # El inicio aumenta en samplesPerThreads    
        start = start + samplesPerThreads
        # El final aumenta en samplesPerThreads
        end = end + samplesPerThreads
        i = i+1
    y = []
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

def signalOperator(start, end, signal, t ,fs, i, fileName, carrier1, carrier2):
    y=[]
    A = 10
    f1=(fs+2000)/4 
    f2=(fs-2000)/4 
    # Se divide la senial de acuerdo al inicio y al final
    signal = signal[start:end]
    
    for sample in signal:
        sample="{0:08b}".format(sample)
        for bit in sample:
            if (bit=="1"):
                y.extend(carrier1)
            else:
                y.extend(carrier2)
    #Se agrega ruido gausiano a la señal modulada
    mean = 0
    std = 1
    #noise = np.random.normal(0.0, A, len(y))
    #y = y + noise
    # Se escribe en un archivo de salida la seccion leida
    sin.writeWav(fileName+str(i+1), fs, np.array(y))
    print("Hebra ", i ,"termino")
def FSK(signal, fs, bitRate, threads, title):
    A=10    
    y = []
    f1=(fs+2000)/4 #Hz
    f2=(fs-2000)/4 #Hz
    t=np.arange(0, 1/bitRate, 1 / fs)
    carrier1 = A*np.cos(2*np.pi*f1*t)
    carrier2 = A*np.cos(2*np.pi*f2*t)
    samplesPerThreads = len(signal)//threads
    start = 0
    end = samplesPerThreads
    i = 0
    # Lista de hebras
    threadList = []
    while i<threads:
        # Se inicializa la hebra
        mythread = threading.Thread(target=signalOperator, args=(start, end, signal, t, fs, i, title, carrier1, carrier2)) 
        # Se agrega a la lista
        threadList.append(mythread)
        # Se inicia
        mythread.start()
        # El inicio aumenta en samplesPerThreads    
        start = start + samplesPerThreads
        # El final aumenta en samplesPerThreads
        end = end + samplesPerThreads
        i = i+1
    """
    for sample in signal:
        sample="{0:08b}".format(sample)
        for bit in sample:
            if (bit=="1"):
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
    noise = np.random.normal(0.0, 1, len(y))
    y = y + noise
    plt.figure(2)
    dCurve=genDigitalCurve(signal, fs, bitRate)
    plt.title("Señal Digital")
    plt.subplot(2,1,1)
    plt.plot(y)
    plt.title(title)    
    plt.subplot(2,1,2)
    plt.plot(dCurve)
    plt.subplots_adjust(hspace = 1)
    
    #fsk_demodulation(y,carrier1,carrier2,t,fs,bitRate)
    """
    return np.array(y)

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
    corr1 = sg.fftconvolve(signal,carrier_1,'same')
    corr2 = sg.fftconvolve(signal,carrier_2,'same')
    plt.figure(5)
    plt.subplot(3,1,1)
    plt.plot(t,carrier_1)
    plt.subplot(3,1,2)
    plt.plot(t,carrier_2)

    #Se obtienen las correlaciones
    #corr1 = sg.medfilt(np.abs(corr1))
    #corr1 = sg.medfilt(np.abs(corr1))
    #corr2 = sg.medfilt(np.abs(corr2))
    #corr2 = sg.medfilt(np.abs(corr2))
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
        return 1
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
        """
        #signal=signal[0:1000]
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
        """
        fs, signal = wavfile.read("../../Wav/"+fileName+".wav")
        test=w.open("../../Wav/"+fileName+".wav", "rb")
        test=test.readframes(test.getnframes())
    else:
        test=[0,1,0,0,0,1]*1000
        plt.figure(4)
        plt.subplot(3,1,1)
        plt.plot(test)
        plt.title("test")
        #plt.subplot(3,1,2)
        #plt.plot(t,carrier_2)
        #plt.title("Carrier 2")
        #plt.subplots_adjust(hspace = 1)
        fs = 1000 #Frecuencia de muestreo en Hz
    fs=10000
    bitRate=100 #Bit por segundo
    threads = int(input("Ingrese cantidad de hebras: "))
    y=np.array([])    
    if(modType=="ASK"):
        #Funcion que modula
        y=ASK(test, fs, bitRate,threads,  title="ASK "+fileName)

    #FSK
    if(modType=="FSK"):
        #Funcion que modula
        print("COMENZANDO MODULACION")
        y=FSK(test, fs, bitRate, threads, title="FSK "+fileName)
    #sin.writeWav(fileName+modType, fs, y)
    plt.show()
    #Plot de correlacionadores
    #demodulacion FSK
    #DDemodulation.mainDigitalDemodulation(flag, bitRate, fileName+modType)

mainDigitalModulation("ASK",1,"handel")