import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import audio2numpy as a2n

class Voice:
    def __init__(self,IsDatabase,fs = 44100):
        self.path = ""
        self.fft = None
        self.freq = None
        self.magnitue = None
        self.phase = None
        self.data = None
        self.time_axis = None
        self.sound_axis = None
        self.specgram = None        #frequencies, times, Pxx
        self.isDatabase = IsDatabase
        self.fs = fs

    def voiceInitializer(self,path):
        self.path = path
        self.data, self.fs = a2n.audio_from_file(path)
        self.time_axis = self.time_axis = np.linspace(0, len(self.data) / self.fs, len(self.data), endpoint=False)
        # self.sound_axis = np.array([self.data[i][0] for i in range(len(self.data))])
        self.fft = np.fft.fft(self.data)
        self.magnitue = np.abs(self.fft)
        self.phase = np.exp((np.angle(self.fft))*1j )
        self.freq = np.fft.fftfreq(len(self.data),1/self.fs)
        self.specgram = spectrogram(self.data, self.fs)

