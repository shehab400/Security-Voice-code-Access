from PyQt5.QtWidgets import QLabel,QRubberBand,QApplication,QMainWindow,QVBoxLayout,QPushButton,QWidget,QErrorMessage,QMessageBox,QDialog,QScrollBar,QSlider, QFileDialog ,QGraphicsView, QGraphicsScene, QGraphicsRectItem, QGraphicsPixmapItem, QVBoxLayout, QWidget, QPushButton, QColorDialog
from PyQt5 import QtWidgets, uic, QtCore,QtGui
from PyQt5.QtCore import QThread,QObject,pyqtSignal as Signal, pyqtSlot as Slot,  Qt,QRect
import sys
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
import sklearn
import librosa

class MyWindow(QMainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
        self.ui = uic.loadUi("GUI.ui", self)
        self.setWindowTitle('Voice Access')
        self.RecordBtn = self.ui.pushButton
        self.RecordBtn.clicked.connect(self.StartRecording)
        self.ui.radioButton.toggled.connect(self.setMode)
        self.ui.radioButton_2.toggled.connect(self.setMode)
        self.isRecording = False
        self.textLabel = self.ui.label_2
        self.sampleVoice = Voice(False)
         # Create Matplotlib figure and axes
        self.matplotlib_figure, self.matplotlib_axes = plt.subplots()
        self.matplotlib_axes.set_axis_off()  # Turn off axes for spectrogram
        # Create Matplotlib widget to embed in PyQT layout
        self.matplotlib_widget = FigureCanvasQTAgg(self.matplotlib_figure)
        self.matplotlib_axes.set_facecolor('white')
        self.matplotlib_figure.patch.set_facecolor('white')
        self.ui.frame_19.layout().addWidget(self.matplotlib_widget)

        self.database = {"Person1": [],"Person2": [],"Person3": [],"Person4": [],"Person5": [],"Person6": [],"Person7": [],"Person8": []}
        for i in range(8):
            self.database[f"Person{i+1}"].append({"Open": Voice(True),"Unlock": Voice(True),"Grant": Voice(True)})
        self.mode = 1

    def setMode(self):
        if self.ui.radioButton.isChecked() == True:
            self.mode = 1
        elif self.ui.radioButton_2.isChecked() == True:
            self.mode = 2

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
        self.generate_spectrogram(self.sampleVoice.specgram)
        self.Comparison()

    def generate_spectrogram(self, specgram):
        frequencies, times, Pxx = specgram
        self.matplotlib_axes.clear()
        # Plot the spectrogram in the Matplotlib figure
        self.matplotlib_axes.pcolormesh(times, frequencies, 10 * np.log10(Pxx), shading='auto', cmap='viridis')
        self.matplotlib_axes.set_xlabel('Time (s)')
        self.matplotlib_axes.set_ylabel('Frequency (Hz)')
        self.matplotlib_axes.set_title('Spectrogram')
        self.matplotlib_axes.set_aspect('auto')
        self.matplotlib_widget.draw()

    def Comparison(self):
        if self.mode == 1:
            self.recognizerSentence()
        elif self.mode == 2:
            self.recognizeSpeaker()



    def recognizeSpeaker(self):
        AUDIO_FILE_PATH = 'Audio Files'
        # A dictionary which stores dominant frequency values for each speaker
        # The keys are student codes
        # ALL Dominant frequency values for each student
        dom_freq_vals = {1: [], 2: [], 3: []}
        # Dictionary that stores the avg feature vector for each speaker in the traininig phase (10)
        feat_vector = {1: 0, 2: 0, 3: 0}
        # Dictionary that stores the 5 feature vectors for each speaker in the testing phase
        feat_vector_test = {1: [], 2: [], 3: []}
        # list of all test vectors for all speakers
        test_vector_list = []
        
        def Training(speaker,flag,number=10,filename = None,trainingNum = 30):
            sentences = ["open middle door ","grant me access ","unlock the gate "]
            if flag == 2:
                sentences = [""]
            for sentence in sentences:
                for i in range(1, number+1):
                    # Create a string which represents the file path
                    path = os.path.join(AUDIO_FILE_PATH,str(speaker))
                    if (filename == None):
                        file_name = f"{sentence}{i}.wav"
                        file_path = os.path.join(path, file_name)
                    else:
                        file_name = filename
                        file_path = file_name

                    # y is a 2d array which stores signal magnitudes of each sample and
                    # number of channels in the audio (by default mono)
                    # sr is the sampling rate (by default sr=22050)
                    y, sr = librosa.load(file_path)

                    # Perform FFT to convert time-domain signal to frequency-domain
                    fft_result = np.fft.fft(y)
                    fft_result_magnitude = np.abs(fft_result)

                    # Generate frequency axis in Hz
                    frequency_axis = np.fft.fftfreq(len(y), 1/sr)

                    # Select positive frequencies and corresponding magnitudes
                    positive_frequencies = frequency_axis[:len(y)//2]
                    positive_magnitudes = fft_result_magnitude[:len(y)//2]

                    # Find the index of the frequency with maximum magnitude (dominant frequency)
                    dominant_frequency_index = np.argmax(positive_magnitudes)

                    # Get the dominant frequency
                    # round to 4 significant digits
                    dominant_frequency = round(positive_frequencies[dominant_frequency_index], 4)
                    dom_freq_vals[speaker].append(dominant_frequency)

            # Get the average feature vector for the speaker
            # Use only the first 10 values to get calculate feature vector (training)
            # round to 4 significant digits
            if flag == 1:
                feat_vector[speaker] = round(np.average(dom_freq_vals[speaker][:trainingNum]), 4)
            elif flag == 2:
                # Get the features for the test vectors (last 5 values)
                feat_vector_test[speaker] = dom_freq_vals[speaker][trainingNum:]
                # Append the test vectors to the list
                test_vector_list.extend(dom_freq_vals[speaker][trainingNum:])
        Training(1,1)
        Training(2,1)
        Training(3,1)
        Training(2,2,1,self.sampleVoice.path)
        # list of avg feature vectors (training)
        feat_vector_list = [feat_vector[val] for val in feat_vector]

        closest_match_vals = {1: [], 2: [], 3: []}
        # Loop over the avg test vectors toget the closest matching indices
        max = 0
        for index, test in enumerate(test_vector_list):
            closest_value = min(feat_vector_list, key=lambda x: abs(x - test))
            closest_index = feat_vector_list.index(closest_value)
            # Check if there has already been a match
            sim1 = round((1-(abs(test-feat_vector[1]))/feat_vector[1])*100,2)
            print(f"For test {index + 1} the recognized student was Person 1\nSimilarity = {sim1}%")
            self.ui.progressBar_16.setValue(int(sim1))
            closest_match_vals[1].append(sim1)
            if sim1 > max:
                max = sim1
                person = "Person 1"

            sim2 = round((1-(abs(test-feat_vector[2]))/feat_vector[2])*100,2)
            print(f"For test {index + 1} the recognized student was Person 2\nSimilarity = {sim2}%")
            self.ui.progressBar_17.setValue(int(sim2))
            closest_match_vals[2].append(sim2)
            if sim2 > max:
                max = sim2
                person = "Person 2"

            sim3 = round((1-(abs(test-feat_vector[3]))/feat_vector[3])*100,2)
            if sim3 < 0:
                sim3 = 3
            print(f"For test {index + 1} the recognized student was Person 3\nSimilarity = {sim3}%")
            self.ui.progressBar_18.setValue(int(sim3))
            closest_match_vals[3].append(sim3)
            if sim3 > max:
                max = sim3
                person = "Person 3"

        if self.ui.comboBox.currentText() == person:
            self.textLabel.setText("ACESS GRANTED😁")
        else:
            self.textLabel.setText("ACESS DENIED😢")
                # print(f"For test {index + 1} the recognized student was Person 3")
                # closest_match_vals[3].append(test)
                    