import sys
sys.path.insert(0, 'Files InOut')
import fileDirector
sys.path.insert(0, 'FFT')
import FFT



#Entradas
print ("***MENU***")
fileName=input("Ingrese el nombre del archivo (sin .wav): ")



#Lectura de .wav
arrayAux = fileDirector.openWav(fileName)
if(len(arrayAux)==0):
    exit()
fs_rate=arrayAux[0]
signal=arrayAux[1]


FFT.calculateFFT(fs_rate, signal)
