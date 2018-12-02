import sys
sys.path.insert(0, 'Files InOut')
import fileDirector
sys.path.insert(0, 'SoundInterface')
import SoundIn as sin
sys.path.insert(0, 'FFT')
import FFT
import threading
from scipy.io import wavfile
from tkinter import *

from tkinter import messagebox

def recordButtonAux(recordFileName, recordSeconds, text1):
    global record_thread
    record_thread = threading.Thread(target=
        lambda:sin.microphone(recordFileName, recordSeconds, text1))
    record_thread.daemon = True
    record_thread.start()
    
def record():
    #Creacion de la ventana
    recordWindow = Toplevel(master)
    #Text
    text1=StringVar()
    text1.set("Esperando...")
    #Labels
    Label(recordWindow, text="Nombre de archivo de grabacion: (Sin .wav)").grid(row=0)
    Label(recordWindow, text="Cantidad de segundos a grabar:").grid(row=1)
    Label(recordWindow, text="Segundos restantes:").grid(row=2, column=2)
    labelSeconds=Label(recordWindow, textvariable=text1)
    labelSeconds.grid(row=2, column=3)
    #Inputs
    fileNameRecord = Entry(recordWindow)
    recordSeconds = Entry(recordWindow)
    fileNameRecord.grid(row=0, column=1)
    recordSeconds.grid(row=1, column=1)
    #Botones
    Button(recordWindow, text = "Comenzar grabación", command = lambda: recordButtonAux(fileNameRecord.get(), int(recordSeconds.get()), text1)).grid(row=2,column=0, sticky=W, pady=4)
    Button(recordWindow, text = "Atrás", command = recordWindow.destroy).grid(row=2,column=1, sticky=W, pady=4)

def fourierFile(fileName, text1, fButton2, fButton3):
    arrayAux = fileDirector.openWav(fileName.get())
    global fs_rate 
    global signal
    if(len(arrayAux)==0):
        messagebox.showinfo("Error", "Archivo de entrada no existe")
        fButton2['state']='disabled'
        fButton3['state']='disabled'
    else:
        fs_rate=arrayAux[0]
        signal=arrayAux[1]
        text1.set("Archivo actual: "+fileName.get()+".wav")    
        fButton2['state']='normal'
        fButton3['state']='normal'

def fourier():
    #Creacion de la ventana
    fourierWindow = Toplevel()
    #Text
    text1=StringVar()
    text1.set("Archivo actual: Ninguno")
    #Labels
    Label(fourierWindow, text="Nombre de archivo: (Sin .wav)").grid(row=0)
    Label(fourierWindow, textvariable=text1).grid(row=1, column=1)
    fileName = Entry(fourierWindow)
    fileName.grid(row=0, column=1)
    Label(fourierWindow, text="Frequencia a modular: ").grid(row=1)
    freq = Entry(fourierWindow)
    freq.grid(row=1, column=1)
    var1 = IntVar()
    Checkbutton(fourierWindow, text="Utilizar coseno como señal", variable=var1).grid(row=3, sticky=W)
    fourierButton2=Button(fourierWindow, state=DISABLED, text = "Modulacion AM", 
    #Botones
        command = lambda: FFT.amModulation(fs_rate, signal, int(freq.get()),var1.get()))
    fourierButton2.grid(row=4, column=0, sticky=W, pady=4)
    fourierButton3=Button(fourierWindow, state=DISABLED, text = "Modulacion FM", 
        command = lambda: FFT.fmModulation(fs_rate, signal, int(freq.get()), var1.get()))
    fourierButton3.grid(row=4, column=1, sticky=W, pady=4)    
    Button(fourierWindow, text = "Abrir archivo", 
        command = lambda: fourierFile(fileName, text1, fourierButton2, fourierButton3)).grid(row=2, column=0, sticky=W, pady=4)


master = Tk()

recordButton = Button(master, text="Grabar", command = record).grid(row=0, column=0, sticky=W, pady=4)
fourierButton = Button(master, text="Transformada", command = fourier).grid(row=0, column=1, sticky=W, pady=4)

master.mainloop()

