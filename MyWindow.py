from PyQt5.QtWidgets import QLabel,QRubberBand,QApplication,QMainWindow,QVBoxLayout,QPushButton,QWidget,QErrorMessage,QMessageBox,QDialog,QScrollBar,QSlider, QFileDialog ,QGraphicsView, QGraphicsScene, QGraphicsRectItem, QGraphicsPixmapItem, QVBoxLayout, QWidget, QPushButton, QColorDialog
from PyQt5 import QtWidgets, uic, QtCore,QtGui
from PyQt5.QtCore import QThread,QObject,pyqtSignal as Signal, pyqtSlot as Slot,  Qt,QRect
import sys
from MyButton import *
import sounddevice as sd
from scipy.io.wavfile import write
import wavio as wv
import os
import wave
import pyaudio
import time
import threading
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from Voice import *

class MyWindow(QMainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
        self.ui = uic.loadUi("GUI.ui", self)
        self.setWindowTitle('Voice Access')
        self.RecordBtn = self.ui.pushButton
        self.RecordBtn.clicked.connect(self.StartRecording)
        self.isRecording = False
        self.textLabel = self.ui.label_2
        self.sampleVoice = Voice(False)
         # Create Matplotlib figure and axes
        self.matplotlib_figure, self.matplotlib_axes = plt.subplots()
        self.matplotlib_axes.set_axis_off()  # Turn off axes for spectrogram
        # Create Matplotlib widget to embed in PyQT layout
        self.matplotlib_widget = FigureCanvasQTAgg(self.matplotlib_figure)
        self.matplotlib_axes.set_facecolor('black')
        self.matplotlib_figure.patch.set_facecolor('black')
        self.ui.frame_19.layout().addWidget(self.matplotlib_widget)

    def StartRecording(self):
        if self.isRecording == False:
            self.isRecording = True
            threading.Thread(target=self.record).start()
        else:
            self.isRecording = False
            
    def record(self):
        audio = pyaudio.PyAudio()
        stream = audio.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)
        frames = []
        start = time.time()

        for i in range(0, int(44100/1024 * 4)):
            data = stream.read(1024)
            frames.append(data)
            passed = time.time() - start
            secs = passed % 60
            mins = passed // 60
            hours = mins // 60
            self.textLabel.setText(f"{int(hours):02d}:{int(mins):02d}:{int(4-secs):02d}")
        
        stream.stop_stream()
        stream.close()
        audio.terminate()
        self.isRecording = False

        exists = True
        i = 1
        while exists:
            if os.path.exists(f"recording{i}.wav"):
                i += 1
            else:
                exists = False

        sound_file = wave.open(f"recording{i}.wav","wb")
        sound_file.setnchannels(1)
        sound_file.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        sound_file.setframerate(44100)
        sound_file.writeframes(b"".join(frames))
        sound_file.close()
        self.textLabel.setText("Loading...")
        self.sampleVoice.voiceInitializer(f"recording{i}.wav")
        self.generate_spectrogram(self.sampleVoice.data,self.sampleVoice.fs)

    def generate_spectrogram(self, data, fs):
        frequencies, times, Pxx = spectrogram(data, fs)
        self.matplotlib_axes.clear()
        # Plot the spectrogram in the Matplotlib figure
        self.matplotlib_axes.pcolormesh(times, frequencies, 10 * np.log10(Pxx), shading='auto', cmap='viridis')
        self.matplotlib_axes.set_xlabel('Time (s)')
        self.matplotlib_axes.set_ylabel('Frequency (Hz)')
        self.matplotlib_axes.set_title('Spectrogram')
        self.matplotlib_axes.set_aspect('auto')
        self.matplotlib_widget.draw()

        