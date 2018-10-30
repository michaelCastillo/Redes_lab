import sys
sys.path.insert(0, 'Files InOut')
import fileDirector
sys.path.insert(0, 'SoundInterface')
import SoundIn
sys.path.insert(0, 'FFT')
import FFT
from scipy.io import wavfile





#Entradas
print ("***MENU***")
fileName=input("Ingrese el nombre del archivo (sin .wav): ")
flag=input("Desea graficos? (1 SI, 0 NO): ")


#Lectura de .wav
arrayAux = fileDirector.openWav(fileName)
samplingFrequency, signalData = fileDirector.openWavToFreqTime(fileName)

if(len(arrayAux)==0):
    exit()
fs_rate=arrayAux[0]
print("fs" + str(fs_rate))
signal=arrayAux[1]

SoundIn.microphone()

#Calcular FFT
#FFT.calculateFFT(fs_rate, signal, flag)
#FFT.getFrequencyTimePlot(samplingFrequency,signalData)
