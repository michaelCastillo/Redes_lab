import numpy as np
import sys
from matplotlib import pyplot as plt
import scipy.io.wavfile as wavfile
sys.path.insert(0, '../FFT')
import FFT as fft_own
from scipy import signal as sg


##importar del archivo.
def getSignalTime(fs_rate, signal):
    signal_len = float(len(signal))
    tAux = float(signal_len) / float(fs_rate)
    t = np.linspace(0, tAux, signal_len)
    return t



def depureMachine(digitalSignal,digitalDemodulation):
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
        return float(errors)*100/float(len(digitalSignal))
    return 0


def fsk_demodulation(signal,f1,f2,fs,bitRate):
    plot = False
    A=100
    t=np.arange(0, 1/bitRate, 1 / fs)
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
        plt.plot(corr1[0:10000])
        plt.subplot(2,1,2)
        plt.plot(corr2[0:10000])
    # plt.subplot(3,1,2)
    # oPlot.plotTransform(xfft_corr2, fftMod_corr2, "Señal Portadora")
    print("Transformadas calculadas")
    
    
    arrayBits = []
    

    
    
    #Se genera un array vacio para almacenar los bits obtenidos.
    #Para recorrer el arreglo de correlaciones se debe mover fs_rate*tiempoBit para encontrar
    # cada maximo
    
    skip = fs//bitRate  #muestras por 1 bit.
    bit_index = skip//2        # Indice del bit inicia en el centro de la primera señal.
    depur = 0
    indexDemod1 = []
    indexDemod2 = []
    while(bit_index < len(corr1)):
        bitCorr1 = corr1[bit_index]
        bitCorr2 = corr2[bit_index]
        if( bitCorr1 > bitCorr2):
            arrayBits.append(1)
        else:
            arrayBits.append(0)
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
