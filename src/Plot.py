
from matplotlib import pyplot as plt
import sys

#GRÁFICA DE LA TRANSFORMADA DE FOURIER
def plotTransform(xft, ft, title):
    #Titulo gráfico
    plt.title("Transformada " + title)
    #Ejes x e y
    plt.xlabel("Frecuencia [Hz]")
    plt.ylabel("Amplitud [dB]")
    plt.plot(xft, abs(ft))


#GRAFICA DE SEÑAL VS TIEMPO
def plotSignalTime(signal, t, title, dot):
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




##################################
# Espectrograma
##################################
def plotSpec(signal,fs_rate,title):
    plt.title("Espectrograma "+title)
    plt.specgram(signal,Fs=fs_rate)