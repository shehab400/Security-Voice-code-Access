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
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
import sklearn.metrics as metrics
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

        self.checkBoxes = [self.ui.checkBox,self.ui.checkBox_2,self.ui.checkBox_3,self.ui.checkBox_4,self.ui.checkBox_5,self.ui.checkBox_6,self.ui.checkBox_7,self.ui.checkBox_8]
        self.database = {"Person1": [],"Person2": [],"Person3": [],"Person4": [],"Person5": [],"Person6": [],"Person7": [],"Person8": []}
        # for i in range(8):
        #     self.database[f"Person{i+1}"].append({"Open": Voice(True),"Unlock": Voice(True),"Grant": Voice(True)})
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
            self.recognizeSentence()
        elif self.mode == 2:
            self.recognizeSpeaker()

    def recognizeSentence(self):
        def extract_features_labels(labelnum,sentence,filename=None,number=10):
            speakers = [1,2,3]
            feature = []
            label = []
            directory_path = 'Audio Files'
            # if sentence == "":
            #     speakers = ['random']
            for speaker in speakers:
                for i in range(1,number+1):
                    # Assuming audio files are in WAV format
                    
                    file_name = f"{speaker}/{sentence}{i}.wav"
                    file_path = os.path.join(directory_path, file_name)
                    # Load the audio file using librosa
                    audio_signal, sample_rate = librosa.load(file_path)
                    
                    # Extract MFCCs from the audio signal
                    mfccs = librosa.feature.mfcc(y=audio_signal, sr=sample_rate, n_mfcc=20)

                    # Compute the mean of MFCCs as features
                    mfccs_mean = np.mean(mfccs, axis=1)
                    
                    # Append MFCCs and associated label (if available) to lists
                    feature.append(mfccs_mean)
                    
                    # Include logic to add associated labels if available (e.g., sentiment labels)
                    label.append(labelnum)
                    # label_list.append(label)
                    # Replace 'label' with the actual labels from your dataset
                    
                    # label_list.append(label)
                    # Replace 'label' with the actual labels from your dataset
            if filename != None:
                file_name = filename
                audio_signal, sample_rate = librosa.load(file_name)
                mfccs = librosa.feature.mfcc(y=audio_signal, sr=sample_rate, n_mfcc=20)
                mfccs_mean = np.mean(mfccs, axis=1)
                feature.append(mfccs_mean)
                label.append(labelnum)
            return feature,label


        feature_list_open ,label_list_open = extract_features_labels(0,"open middle door ")
        X_open = np.array(feature_list_open)
        y_open = np.array(label_list_open)


        feature_list_grant ,label_list_grant = extract_features_labels(1,"grant me access ",self.sampleVoice.path)
        X_grant = np.array(feature_list_grant)
        y_grant = np.array(label_list_grant)
        

        feature_list_unlock ,label_list_unlock = extract_features_labels(2,"unlock my gate ")
        X_unlock = np.array(feature_list_unlock)
        y_unlock = np.array(label_list_unlock)

        # feature_list_random ,label_list_random = extract_features_labels(3,"",number=24)
        # X_random = np.array(feature_list_random)
        # y_random = np.array(label_list_random)

        X=np.concatenate((X_open, X_grant, X_unlock), axis=0)

        y=np.concatenate((y_open, y_grant, y_unlock), axis=0)

        for l in range(1):
            Xt_open = X_open[:30]
            Xs_open = X_open[30:]
            yt_open = y_open[:30]
            ys_open = y_open[30:]
            Xt_grant = X_grant[:30]
            Xs_grant = X_grant[30:]
            yt_grant = y_grant[:30]
            ys_grant = y_grant[30:]
            Xt_unlock = X_unlock[:30]
            Xs_unlock = X_unlock[30:]
            yt_unlock = y_unlock[:30]
            ys_unlock = y_unlock[30:]
            # Xt_random = X_random[:24]
            # Xs_random = X_random[24:]
            # yt_random = y_random[:24]
            # ys_random = y_random[24:]

            X_train=np.concatenate((Xt_open, Xt_grant, Xt_unlock), axis=0)
            X_test=np.concatenate((Xs_open, Xs_grant, Xs_unlock), axis=0)
            y_train=np.concatenate((yt_open, yt_grant, yt_unlock), axis=0)
            y_test=np.concatenate((ys_open, ys_grant, ys_unlock), axis=0)



            # Train a classifier (Random Forest as an example)
            classifier = RandomForestClassifier(n_estimators = 2000, random_state=1000)
            classifier.fit(X_train, y_train)


            # Predict on test set
            y_pred = classifier.predict(X_test)
            probs = classifier.predict_proba(X_test)
            self.ui.progressBar_13.setValue(int(probs[0][0]*100))
            self.ui.progressBar_14.setValue(int(probs[0][1]*100))
            self.ui.progressBar_15.setValue(int(probs[0][2]*100))
            if y_pred[0] == 0 and probs[0][0] >= 0.4:
                if self.mode == 1:
                    self.textLabel.setText("ACCESS GRANTED😁")
                s = 1
            elif y_pred[0] == 1 and probs[0][1] >= 0.4:
                if self.mode == 1:
                    self.textLabel.setText("ACCESS GRANTED😁")
                s = 2
            elif y_pred[0] == 2 and probs[0][2] >= 0.4:
                if self.mode == 1:
                    self.textLabel.setText("ACCESS GRANTED😁")
                s = 3
            else:
                if self.mode == 1:
                    self.textLabel.setText("ACCESS DENIED😢")
                s = 0
            return s

    def recognizeSpeaker(self):
        AUDIO_FILE_PATH = 'Audio Files'
        # A dictionary which stores dominant frequency values for each speaker
        # The keys are student codes
        # ALL Dominant frequency values for each student
        dom_freq_vals = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: []}
        # Dictionary that stores the avg feature vector for each speaker in the traininig phase (10)
        feat_vector = {1: [0,0,0], 2: [0,0,0], 3: [0,0,0], 4: [0,0,0], 5: [0,0,0], 6: [0,0,0], 7: [0,0,0]}
        # Dictionary that stores the 5 feature vectors for each speaker in the testing phase
        feat_vector_test = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: []}
        # list of all test vectors for all speakers
        test_vector_list = []
        temp = self.recognizeSentence()
        
        def Training(speaker,flag,number=10,filename = None,trainingNum = 30):
            sentences = ["open middle door ","grant me access ","unlock my gate "]
            if flag == 2:
                sentences = [""]
            for sentence,j in zip(sentences,range(3)):
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
                    feat_vector[speaker][j] = round(np.average(dom_freq_vals[speaker][j*10:j*10+10]), 4)
                elif flag == 2:
                    # Get the features for the test vectors (last 5 values)
                    feat_vector_test[speaker] = dom_freq_vals[speaker][trainingNum:]
                    # Append the test vectors to the list
                    test_vector_list.extend(dom_freq_vals[speaker][trainingNum:])


        for i in range(1,8):
            Training(i,1)
        
        # a = np.array((feat_vector[1],feat_vector[2],feat_vector[3],feat_vector[4],feat_vector[5],feat_vector[6]))
        # X_train = np.zeros([len(a),1])
        # for i in range(len(a)):
        #     X_train[i][0] = a[i]
        # a = np.array((1,2,3,4,5,6))
        # y_train = np.zeros([len(a),1])
        # for i in range(len(a)):
        #     y_train[i][0] = a[i]

        Training(2,2,1,self.sampleVoice.path)
        
        # X_test = np.array([feat_vector_test[2]])
        # classifier = RandomForestClassifier(n_estimators= 2000, random_state= 42)
        # classifier.fit(X_train,y_train)
        # y_pred = classifier.predict(X_test)
        # print(classifier.predict_proba(X_test))

        # list of avg feature vectors (training)
        feat_vector_list = [feat_vector[val] for val in feat_vector]
        print (feat_vector_list)
        # Loop over the avg test vectors toget the closest matching indices
        # max = 0
        progressBars = [self.ui.progressBar_16,self.ui.progressBar_17,self.ui.progressBar_18,self.ui.progressBar,self.ui.progressBar_20,self.ui.progressBar_21,self.ui.progressBar_22]
        for index, test in enumerate(test_vector_list):
            print(test)
            for i,progress in zip(range(1,8),progressBars):
                sim = round((1-(abs(test-feat_vector[i][temp-1]))/abs(feat_vector[i][temp-1]))*100,2)
                if sim < 0:
                    sim = 3
                print(f"For test 1 the recognized student was Person {i}\nSimilarity = {sim}%")
                progress.setValue(int(sim))
                # if sim > max:
                #     max = sim
                #     person = i
        if temp != 0:
            for check,progress in zip(self.checkBoxes,progressBars):
                if check.isChecked() == True and progress.value() >= 82:
                    self.textLabel.setText("ACCESS GRANTED😁")
                    return
        self.textLabel.setText("ACCESS DENIED😢")