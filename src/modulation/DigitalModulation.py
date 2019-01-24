import math
import sys
sys.path.insert(0, '../')
import Plot as oPlot
sys.path.insert(0, './')
import DigitalDemodulation as demod
sys.path.insert(0, '../FFT/')
import FFT as fttMod


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
    
    # ask_demodulation(y,carrier1,carrier2,t,fs,bitRate)

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

def signalOperatorQfsk(start, end, signal,i, carrier1, carrier2,carrier3,carrier4,arrayMutex, arrayResult):
    y=[]
    # Se divide la senial de acuerdo al inicio y al final
    signal = signal[start:end]
    
    for sample in signal:
        sample="{0:08b}".format(sample)
        symbolIndex = 1
        buff = ""
        for bit in sample:
            buff = buff + bit
            if (symbolIndex%2 == 0):
                if (buff =="00"):
                    y.extend(carrier1)
                elif (buff=="01"):
                    y.extend(carrier2)
                elif (buff=="10"):
                    y.extend(carrier3)
                else:
                    y.extend(carrier4)   ##11
                buff = ""
            symbolIndex = symbolIndex + 1
    # Se escribe en un archivo de salida la seccion leida
    print("Hebra ", i ,"termino")

def QFSK(signal, fs, bitRate, threads, title):
    
    plot = False
    signalModulated=[]
    A=1
    y = []
    f1 = 6000
    f4 = 2000
    fs = 5*f1
    t=np.arange(0, 1/bitRate, 1 / fs)
    carrier1 = A*np.cos(2*np.pi*f1*t)
    carrier2 = A*np.cos(2*np.pi*f4*t)
    carrier3 = A*np.sin(2*np.pi*f1*t)
    carrier4 = A*np.sin(2*np.pi*f4*t)
    #Se crea la señal que da el inicio de la lectura de datos
    initSignal = createInitSignal(14000)
    y = initSignal
    samplesPerThreads = len(signal)//threads
    start = 0
    end = samplesPerThreads
    i = 0
    # Lista de hebras
    barrier = threading.Barrier(threads)
    threadList = []
    while i<threads:
        # Se inicializa la hebra
        mythread = threading.Thread(target=signalOperatorQfsk, args=(start, end, signal, t, fs, i, title, carrier1, carrier2,carrier3,carrier4,barrier)) 
        # Se agrega a la lista
        threadList.append(mythread)
        # Se inicia
        mythread.start()
        # El inicio aumenta en samplesPerThreads    
        start = start + samplesPerThreads
        # El final aumenta en samplesPerThreads
        end = end + samplesPerThreads
        i = i+1

    
    
    if(plot):
        print("Inicio de ploteo")
        #Grafica de las portadoras
        lenCarriers = len(carrier1)
        lenTime = len(t)
        plt.figure(1)
        plt.subplot(2,2,1)
        oPlot.plotSignalTime(carrier1[0:lenCarriers//8],t[0:lenTime//8],"Portadora (1)",False)
        plt.subplot(2,2,2)
        oPlot.plotSignalTime(carrier2[0:lenCarriers//8],t[0:lenTime//8],"Portadora (0) ",False)
        plt.subplot(2,2,3)
        oPlot.plotSignalTime(carrier3[0:lenCarriers//8],t[0:lenTime//8],"Portadora (1)",False)
        plt.subplot(2,2,4)
        oPlot.plotSignalTime(carrier4[0:lenCarriers//8],t[0:lenTime//8],"Portadora (0) ",False)
        #Grafica de la señal modulada
        plt.figure(2)
        oPlot.plotSignalTime()
    
    #Demodulacion


def QFSKSecuential(signal, fs, bitRate, title,audio):
    
    plot = False
    signalModulated=[]
    A=1
    y = []
    f1 = 10000
    f4 = 5000
    fs = 14000*6
    t=np.arange(0, 1/bitRate, 1 / fs)
    carrier1 = A*np.cos(2*np.pi*f1*t)
    carrier2 = A*np.cos(2*np.pi*f4*t)
    carrier3 = A*np.sin(2*np.pi*f1*t)
    carrier4 = A*np.sin(2*np.pi*f4*t)
    
    #Se crea la señal que da el inicio de la lectura de datos
    initSignal = createInitSignal(14000)
    y = []
    y.extend(initSignal)
    
    buff = ""
    if(not audio):
        print("Procesando audio...")
        for sample in signal:
            sample="{0:08b}".format(sample)
            symbolIndex = 1
            buff = ""
            for bit in sample:
                buff = buff + bit
                if (symbolIndex%2 == 0):
                    if (buff =="00"):
                        y.extend(carrier1)
                    elif (buff=="01"):
                        y.extend(carrier2)
                    elif (buff=="10"):
                        y.extend(carrier3)
                    else:
                        y.extend(carrier4) 
                    buff = ""
                symbolIndex = symbolIndex + 1

    else:
        print("Procesando array de bits...")
        symbolIndex = 1
        for bit in signal:
            buff = buff + str(bit)
            if (symbolIndex%2 == 0):
                #print(buff)
                if (buff =="00"):
                    y.extend(carrier1)
                elif (buff=="01"):
                    y.extend(carrier2)
                elif (buff=="10"):
                    y.extend(carrier3)
                else:
                    y.extend(carrier4)   ##11
                buff = ""
            symbolIndex = symbolIndex + 1
    
    
    if(plot):
        print("Inicio de ploteo")
        #Grafica de las portadoras
        lenCarriers = len(carrier1)
        lenTime = len(t)
        plt.figure(1)
        plt.subplot(2,2,1)
        oPlot.plotSignalTime(carrier1[0:lenCarriers],t[0:lenTime],"Portadora (00)",False)
        plt.subplot(2,2,2)
        oPlot.plotSignalTime(carrier2[0:lenCarriers],t[0:lenTime],"Portadora (01) ",False)
        plt.subplot(2,2,3)
        oPlot.plotSignalTime(carrier3[0:lenCarriers],t[0:lenTime],"Portadora (10)",False)
        plt.subplot(2,2,4)
        oPlot.plotSignalTime(carrier4[0:lenCarriers],t[0:lenTime],"Portadora (11) ",False)

    # plt.figure(2)
    # timeModulated = getSignalTime(fs,y)
    # oPlot.plotSignalTime(y[0:len(signal)//16],timeModulated[0:len(signal)//16],"Senal modulada",False)
    y.extend(initSignal)
    y.extend(initSignal)
    y.extend(initSignal)
    return y
    #Demodulacion


def FSKSeq(signal, fs, bitRate, threads, title):
    
    plot = True
    signalModulated=[]
    A=10    
    f1= 15000
    f2= 4000
    fs = 10*f1
    t=np.arange(0, 1/bitRate, 1 / fs)
    carrier1 = A*np.cos(2*np.pi*f1*t)
    carrier2 = A*np.cos(2*np.pi*f2*t)
    i = 0
    for bit in signal:
        if(bit == 1):
            signalModulated.extend(carrier1)
        else:
            signalModulated.extend(carrier2)
    if(plot):
        #Grafica de las portadoras
        plt.figure(1)
        plt.subplot(2,1,1)
        lenCarriers = len(carrier1)
        oPlot.plotSignalTime(carrier1[0:lenCarriers//2],t[0:lenCarriers//2],"Portadora (1)",False)
        plt.subplot(2,1,2)
        oPlot.plotSignalTime(carrier2[0:lenCarriers//2],t[0:lenCarriers//2],"Portadora (0) ",False)
        #Grafica de la señal modulada

    return signalModulated

def FSK(signal, fs, bitRate, threads, title):
    
    plot = False
    signalModulated=[]
    A=10    
    y = []
    f1= 15000
    f2= 4000
    fs = 5*f1
    t=np.arange(0, 1/bitRate, 1 / fs)
    carrier1 = A*np.cos(2*np.pi*f1*t)
    carrier2 = A*np.cos(2*np.pi*f2*t)
    samplesPerThreads = len(signal)//threads
    start = 0
    end = samplesPerThreads
    i = 0
    barrier = threading.Barrier(threads, timeout = printTimeoutBarrier)
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
    barrier.wait()
    
    if(plot):
        #Grafica de las portadoras
        plt.figure(1)
        plt.subplot(3,1,1)
        lenCarriers = len(carrier1)
        oPlot.plotSignalTime(carrier1[0:lenCarriers//4],t,"Portadora (1)",False)
        plt.subplot(3,1,2)
        oPlot.plotSignalTime(carrier1[0:lenCarriers//4],t,"Portadora (0) ",False)
        #Grafica de la señal modulada

def printTimeoutBarrier():
    print("Se ha agotado el tiempo para que lleguen las threads")


def mainDigitalModulation(modType,flag,fileName):
    flagTest=True
    audioParams = []
    test=[]
    binarySignal=[]
    if(not flagTest):  
        #fs, signal = wavfile.read("../../Wav/"+fileName+".wav")
        fs, signal = FD.openDigitalWav(fileName)
        test=w.open("../../Wav/"+fileName+".wav", "rb")
        audioParams = test.getparams()
        test=test.readframes(test.getnframes())
    else:
        test=np.random.randint(2, size=10000)

    baudRate=2000
    y=np.array([])    
    result = []
    fs = 10
    threads = 0
    if(modType=="ASK"):
        y=ASK(test, fs, baudRate,threads,  title="ASK "+fileName)
    if(modType=="FSK"):
        print("Es fsk")
        y=FSKSeq(test, fs, baudRate, threads, title="FSK "+fileName)
        demod.fsk_demodulation(y,15000,4000,15000*10,baudRate)
    if (modType == "QFSK"):
        flgCompleteTest = False
        if(flgCompleteTest):
            completeTest(baudRate,20)
        else:
            
            print("######### Modulacion #############")
            y = QFSKSecuential(test, fs, baudRate, title="QFSK "+fileName,audio=flagTest)
            noise = np.random.normal(0.0, 1, len(y))
            # y = y + noise
            print("Escribiendo modulacion ...")
            sin.writeWav("newQFSK"+fileName,14000*6,np.array(y))
            print("######### Demodulacion ###########")

            """
            demodulation = demod.qam_demodulation(y,baudRate)
            print("#########   Testing   ############")
            
            if(not flagTest):
                result = demod.depureMachine(test,demodulation)
                print("Tasa de error: "+str(result)+"%")
                print("#########  Fin testing  #########")
            else:
                result = demod.depureMachineSecuential(test,demodulation)
                print("Tasa de error: "+str(result)+"%")
                print("#########  Fin testing  #########")
            print("######### Escribiendo el resultado #########")
            
            #data = parseDataToBytes(demodulation)
            
            writeWav("../../Wav/wavQFSK"+fileName,demodulation,audioParams)
            """
            print("#########   Fin del procesamiento    #######")


    plt.show()
    #Plot de correlacionadores
    #demodulacion FSK
    #DDemodulation.mainDigitalDemodulation(flag, bitRate, fileName+modType)


def completeTest(baudRate, times):
    baudTesting = True
    results = []
    fs = 6000*5
    threads = 10
    fileName = "asd"
    baudRate = 100
    rates = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

    for i in range(0,times):
        if(baudTesting):
            baudRate = baudRate + 100

        test=np.random.randint(2, size=100000)
        print("######### Modulacion #############")
        y = QFSKSecuential(test, fs, baudRate,"QFSK "+fileName,True)
        noise = np.random.normal(0, 1, len(y))
        y = y + noise
        print("######### Demodulacion ###########")
        demodulation = demod.qam_demodulation(y,baudRate)
        print("#########   Testing   ############")
        result = demod.depureMachineSecuential(test,demodulation)
        results.append(result)
        print("bdr: " +str(baudRate)+"  Tasa de error: "+str(result)+"%")
        print("#########  Fin testing  #########")
        
    plt.figure(5)
    plt.plot(rates,results)


def writeWav(path,data,params):
    arrayBytes = parseDataToBytes(data)
    output = w.open(path,'wb')
    print("parametros: "+str(params))

    output.setparams(tuple(params))
    output.writeframesraw(arrayBytes)
    output.close()

def parseDataToBytes(data):

    i = 1
    bytesArray = []
    byteBuff = ""
 
    for bit in data :
        byteBuff = byteBuff + str(bit)
        if( (i%8) == 0):
            #byteValue = convertStringToByte(byteBuff)
            byteValue = int(byteBuff,2)
            bytesArray.append(byteValue)
            byteBuff = ""
        i = i + 1
    #print(bytearray(bytesArray))
    return bytearray(bytesArray)

def convertStringToByte(string):
    string = "0x%x" % (int(string, 2))
    return string.encode()

def BytesToInt(data):
    a = 10

def createInitSignal(freq):
    fs = freq*6
    t=np.arange(0, float(1), 1 / fs)
    initSignal = np.cos(2*np.pi*freq*t)
    return initSignal



result=[]
# mainDigitalModulation("QFSK",1,"mario")
