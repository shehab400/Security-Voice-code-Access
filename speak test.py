import os
import numpy as np
import librosa

AUDIO_FILE_PATH = 'C:/Users/dolla/OneDrive/Documents/GitHub/Security-Voice-code-Access/AudioFiles'
# A dictionary which stores dominant frequency values for each speaker
# The keys are student codes
# ALL Dominant frequency values for each student
dom_freq_vals = {1210148: [], 1210280: [], 1210339: []}
# Dictionary that stores the avg feature vector for each speaker in the traininig phase (10)
feat_vector = {1210148: 0, 1210280: 0, 1210339: 0}
# Dictionary that stores the 5 feature vectors for each speaker in the testing phase
feat_vector_test = {1210148: [], 1210280: [], 1210339: []}
# list of all test vectors for all speakers
test_vector_list = []

def Training(speaker,flag,number=10,filename = None):
            for i in range(1, number+1):
                # Create a string which represents the file path
                #  All files are stored in the format of: <Student_ID>_<Record#>.wav
                if (filename == None):
                    file_name = f"{speaker}_{i}.wav"
                else:
                    file_name = filename
                file_path = os.path.join(AUDIO_FILE_PATH, file_name)

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
                feat_vector[speaker] = round(np.average(dom_freq_vals[speaker][:10]), 4)
            elif flag == 2:
                # Get the features for the test vectors (last 5 values)
                feat_vector_test[speaker] = dom_freq_vals[speaker][10:]
                # Append the test vectors to the list
                test_vector_list.extend(dom_freq_vals[speaker][10:])

Training(1210148,1)
Training(1210148,2,1,f"Recording (6).wav")
# list of avg feature vectors (training)
feat_vector_list = [feat_vector[val] for val in feat_vector]

closest_match_vals = {1210148: [], 1210280: [], 1210339: []}
# Loop over the avg test vectors toget the closest matching indices
for index, test in enumerate(test_vector_list):
    closest_value = min(feat_vector_list, key=lambda x: abs(x - test))
    closest_index = feat_vector_list.index(closest_value)
    # Check if there has already been a match
    if closest_index == 0:
        print(f"For test {index + 1} the recognized student was 1210148\nSimilarity = {(1-(abs(test-feat_vector[1210148]))/feat_vector[1210148])*100}%")
        closest_match_vals[1210148].append(test)
    elif closest_index == 1:
        print(f"For test {index + 1} the recognized student was 1210280")
        closest_match_vals[1210280].append(test)
    elif closest_index == 2:
        print(f"For test {index + 1} the recognized student was 1210339")
        closest_match_vals[1210339].append(test)