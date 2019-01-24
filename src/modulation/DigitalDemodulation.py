import numpy as np
import sys
from matplotlib import pyplot as plt
import scipy.io.wavfile as wavfile
sys.path.insert(0, '../FFT')
import FFT as fft_own
from scipy import signal as sg
sys.path.insert(0, '../')
import Plot as oPlot

##importar del archivo.
def getSignalTime(fs_rate, signal):
    signal_len = float(len(signal))
    tAux = float(signal_len) / float(fs_rate)
    t = np.linspace(0, tAux, signal_len)
    return t



def depureMachine(digitalSignal,digitalDemodulation):
    print("tamano audio: "+str(len(digitalSignal)) + "  demod: "+str(len(digitalDemodulation)))
    errors = 0
    i = 0
    for byte in digitalSignal:
        arrayBits = "{0:08b}".format(byte)
        for bit in arrayBits:
            bit = int(bit)
            if(bit != digitalDemodulation[i]):
                errors = errors + 1
            i = i + 1
    if(errors != 0):
        return float(errors)*100/float(len(digitalSignal)*8)
    return 0

def depureMachineSecuential(digitalSignal,digitalDemodulation):
    print("lens => "+str(len(digitalSignal))+"  "+str(len(digitalDemodulation)))
    errors = 0
    i = 0
    for bit in digitalSignal:
        # print(str(i))
        if(bit != digitalDemodulation[i]):
            errors = errors + 1
        i = i + 1
    print("Errores: =>" + str(errors))
    if(errors != 0):
        return float(errors)*100/float(len(digitalSignal))
    return 0

    




def fsk_demodulation(signal,f1,f2,fs,bitRate):
    plot = True
    A=1
    print("len signal => ",str(len(signal)))
    t=np.arange(0, 1/bitRate, 1 / fs)
    print("f1=> ",f1)
    print("f2=> ",f2)
    carrier1 = A*np.cos(2*np.pi*f1*t)
    carrier2 = A*np.cos(2*np.pi*f2*t)

    corr1 = np.correlate(signal,carrier1,'same')
    corr2 = np.correlate(signal,carrier2,'same')

    corr1 = np.abs(corr1)
    corr2 = np.abs(corr2)
    print("He realizado las correlaciones! Grafico.")
    if(plot):
        plt.figure(2)
        plt.subplot(2,1,1)
        plt.plot(corr1[0:len(corr1)//100])
        plt.subplot(2,1,2)
        plt.plot(corr2[0:len(corr1)//100])
    # plt.subplot(3,1,2)
    # oPlot.plotTransform(xfft_corr2, fftMod_corr2, "Señal Portadora")
    print("Transformadas calculadas")
    
    
    arrayBits = []

    return arrayBits



def qam_demodulation(signal,baudrate):

    plot = False
    carriers = False
    
    A=1
    f1 = 10000
    f4 = 5000
    fs = 14000*6 ##Debido a la senal de inicio
    t=np.arange(0, 1/baudrate, 1 / fs)
    carrier1 = A*np.cos(2*np.pi*f1*t)
    carrier2 = A*np.cos(2*np.pi*f4*t)
    carrier3 = A*np.sin(2*np.pi*f1*t)
    carrier4 = A*np.sin(2*np.pi*f4*t)
    #signal = signal[0:len(signal)//8]
    
    
    corr1 = np.correlate(signal,carrier1,'same')
    corr2 = np.correlate(signal,carrier2,'same')
    corr3 = np.correlate(signal,carrier3,'same')
    corr4 = np.correlate(signal,carrier4,'same')

    corr1 = np.abs(corr1)
    corr2 = np.abs(corr2)
    corr3 = np.abs(corr3)
    corr4 = np.abs(corr4)
    if(plot):
        print("He realizado las correlaciones! Grafico.")
        plt.figure(3)
        plt.subplot(2,2,1)
        plt.plot(corr1[0:(len(corr1))//500],label="Corr 00")
        plt.subplot(2,2,2)
        plt.plot(corr2[0:(len(corr1))//500],label="Corr 01")
        plt.subplot(2,2,3)
        plt.plot(corr3[0:(len(corr1))//500],label="Corr 10")
        plt.subplot(2,2,4)
        plt.plot(corr4[0:(len(corr1))//500],label="Corr 11")
    if(carriers):
        plt.figure(4)
        plt.subplot(2,2,1)
        oPlot.plotSignalTime(carrier1,t,"Senal en el tiempo",False)
        plt.subplot(2,2,2)
        oPlot.plotSignalTime(carrier2,t,"Senal en el tiempo",False)
        plt.subplot(2,2,3)
        oPlot.plotSignalTime(carrier3,t,"Senal en el tiempo",False)
        plt.subplot(2,2,4)
        oPlot.plotSignalTime(carrier4,t,"Senal en el tiempo",False)
    print("Transformadas calculadas")
    
    
    arrayBits = []

    #Se genera un array vacio para almacenar los bits obtenidos.
    #Para recorrer el arreglo de correlaciones se debe mover fs_rate*tiempoBit para encontrar
    # cada maximo
    
    skip = fs//baudrate  #muestras por 1 bit.
    bit_index = skip//2        # Indice del bit inicia en el centro de la primera señal.
    while(bit_index < len(corr1)):
        bitCorr1 = corr1[bit_index]
        bitCorr2 = corr2[bit_index]
        bitCorr3 = corr3[bit_index]
        bitCorr4 = corr4[bit_index]

        op_corr1 = (bitCorr1 > bitCorr2) and (bitCorr1 > bitCorr3) and (bitCorr1 > bitCorr4)
        op_corr2 = (bitCorr2 > bitCorr3) and (bitCorr2 > bitCorr4) and (bitCorr2 > bitCorr1)
        op_corr3 = (bitCorr3 > bitCorr2) and (bitCorr3 > bitCorr1) and (bitCorr3 > bitCorr4)
        op_corr4 = (bitCorr4 > bitCorr1) and (bitCorr4 > bitCorr2) and (bitCorr4 > bitCorr3)

        if(op_corr1):
            arrayBits.append(0)
            arrayBits.append(0)
        elif(op_corr2):
            arrayBits.append(0)
            arrayBits.append(1)
        elif(op_corr3):
            arrayBits.append(1)
            arrayBits.append(0)
        else:
            arrayBits.append(1)
            arrayBits.append(1)
            
        bit_index = bit_index + skip
    return arrayBits



def ask_demodulation(signal,carrier_1,carrier_2,t,fs_rate,bitRate):
    signalTime = getSignalTime(fs_rate,signal)
    corr1 = sg.fftconvolve(signal,carrier_1,'same')
    # Se obtienen las correlaciones
    corr1 = sg.medfilt(np.abs(corr1))
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

    dCurve= genDigitalCurve(arrayBits, fs_rate, bitRate)
    result = depureMachine([0,1,0,0,0,1,]*10,arrayBits)

    if(result == 0):
        print("Demodulacion exitosa")
    else:
        print("Demodulacion fallida")


    