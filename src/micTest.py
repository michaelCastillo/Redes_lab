
import pyaudio
import numpy as np
import time
import wave
import matplotlib.pyplot as plt
import time

# open stream
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

CHUNK = 2048 # RATE / number of updates per second

RECORD_SECONDS = 200


# use a Blackman window
window = np.blackman(CHUNK)

x = 0

def listener(stream):
    t1=time.time()
    data = stream.read(CHUNK, exception_on_overflow=False)
    waveData = wave.struct.unpack("%dh"%(CHUNK), data)
    npArrayData = np.array(waveData)
    indata = npArrayData*window
    #Plot time domain
    """
    ax1.cla()
    ax1.plot(indata)
    ax1.grid()
    ax1.axis([0,CHUNK,-5000,5000])
    """
    fftData=np.abs(np.fft.rfft(indata))
    fftTime=np.fft.rfftfreq(CHUNK, 1./RATE)
    which = fftData[1:].argmax() + 1
    #Plot frequency domain graph
    
    #ax2.cla()
    #ax2.plot(fftTime,fftData)
    #ax2.grid()
    #ax2.axis([0,20000,0,10**6])
    #plt.pause(0.00000000000000001)

    #print("took %.02f ms"%((time.time()-t1)*1000))
    # use quadratic interpolation around the max
    if which != len(fftData)-1:
        y0,y1,y2 = np.log(fftData[which-1:which+2:])
        x1 = (y2 - y0) * .5 / (2 * y1 - y2 - y0)
        # find the frequency and output it
        thefreq = (which+x1)*RATE/CHUNK
        print ("The freq is %f Hz." % (thefreq))
    else:
        thefreq = which*RATE/CHUNK
        cprint ("The freq is %f Hz." % (thefreq))
def player():
    i = 1
    p=pyaudio.PyAudio()
    while True:
        stream=p.open(format=pyaudio.paInt16,channels=1,rate=RATE,input=True,
                  frames_per_buffer=CHUNK)
        wf = wave.open("../../Wav/mario"+".wav", 'rb')
        stream = p.open(format =
                    p.get_format_from_width(wf.getsampwidth()),
                    channels = wf.getnchannels(),
                    rate = wf.getframerate(),
                    output = True)
                    
        # read data (based on the chunk size)
        data = wf.readframes(CHUNK)
        while data != b'':
            # writing to the stream is what *actually* plays the sound.
            stream.write(data)
            data = wf.readframes(CHUNK)
            print(data)
        stream.close()    

        plt.ion()
        #fig = plt.figure(figsize=(10,8))
        #ax1 = fig.add_subplot(211)
        #ax2 = fig.add_subplot(212)
        p.terminate()
        p=pyaudio.PyAudio()

        stream2=p.open(format=pyaudio.paInt16,channels=1,rate=RATE,input=True,
                  frames_per_buffer=CHUNK)
        timeout = time.time() + 2
        while time.time() < timeout:
            soundPlot(stream2)
        stream2.stop_stream()
        stream2.close()
        #p.terminate()

        i=i+1

# playTest()
# """ 
p=pyaudio.PyAudio()
stream=p.open(format=pyaudio.paInt16,channels=1,rate=RATE,input=True,
                frames_per_buffer=CHUNK)

plt.ion()
fig = plt.figure(figsize=(10,8))
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
while True:
    soundPlot(stream)

stream.stop_stream()
stream.close()
p.terminate()
# """