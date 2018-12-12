import pyaudio
import wave
from tkinter import *
from tkinter import messagebox
import scipy.io.wavfile as wavfile

def microphone(recordFileName, recordSeconds, text1):
    """PyAudio example: Record a few seconds of audio and save to a WAVE file."""
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    RECORD_SECONDS = recordSeconds
    WAVE_OUTPUT_FILENAME = "../Wav/"+recordFileName+".wav"

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("* recording")

    frames = []
    aux = 0
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        if(i%43==0):
            text1.set(str(RECORD_SECONDS-aux))
            aux=aux+1
        data = stream.read(CHUNK)
        frames.append(data)
    text1.set("Listo!")
    
    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

 