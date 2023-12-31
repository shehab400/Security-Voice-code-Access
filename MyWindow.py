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

class MyWindow(QMainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
        self.RecordBtn = MyButton('Record',self)
        pixmap = QtGui.QPixmap('Images\\Record.png').scaled(15, 15, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self.RecordBtn.setPixmap(pixmap)
        self.RecordBtn.clicked.connect(self.StartRecording)
        self.isRecording = False
        self.textLabel = QLabel("00:00:00",self)
        self.textLabel.move(25,30)

    def StartRecording(self):
        if self.isRecording == False:
            pixmap = QtGui.QPixmap('Images\\StopRecording.png').scaled(18, 18, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            self.RecordBtn.setPixmap(pixmap)
            self.isRecording = True
            threading.Thread(target=self.record).start()
        else:
            pixmap = QtGui.QPixmap('Images\\Record.png').scaled(15, 15, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            self.RecordBtn.setPixmap(pixmap)
            self.isRecording = False
            
    def record(self):
        audio = pyaudio.PyAudio()
        stream = audio.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)
        frames = []
        start = time.time()

        while self.isRecording:
            data = stream.read(1024)
            frames.append(data)

            passed = time.time() - start
            secs = passed % 60
            mins = passed // 60
            hours = mins // 60
            self.textLabel.setText(f"{int(hours):02d}:{int(mins):02d}:{int(secs):02d}")
        
        stream.stop_stream()
        stream.close()
        audio.terminate()

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

        