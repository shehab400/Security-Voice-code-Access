import librosa
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
import sklearn.metrics as metrics
import os


def extract_features_labels(labelnum,sentence,filename=None,number=10):
    speakers = [1]
    feature = []
    label = []
    directory_path = 'Audio Files'
    for speaker in speakers:
        for i in range(1,number+1):
            # Assuming audio files are in WAV format
            
            file_name = f"{speaker}/{sentence}{i}.wav"
            file_path = os.path.join(directory_path, file_name)
            # Load the audio file using librosa
            audio_signal, sample_rate = librosa.load(file_path)
            
            # Extract MFCCs from the audio signal
            mfccs = librosa.feature.mfcc(y=audio_signal, sr=sample_rate, n_mfcc=13)

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
        mfccs = librosa.feature.mfcc(y=audio_signal, sr=sample_rate, n_mfcc=13)
        mfccs_mean = np.mean(mfccs, axis=1)
        feature.append(mfccs_mean)
        label.append(labelnum)
    return feature,label


feature_list_yasar ,label_list_yasar =extract_features_labels(0,"open middle door ")
X_yasar = np.array(feature_list_yasar)
y_yasar = np.array(label_list_yasar)


feature_list_qif ,label_list_qif =extract_features_labels(1,"grant me access ")
X_qif = np.array(feature_list_qif)
y_qif = np.array(label_list_qif)
 

feature_list_taharak ,label_list_taharak =extract_features_labels(2,"unlock the gate ","recording7.wav")
X_taharak = np.array(feature_list_taharak)
y_taharak = np.array(label_list_taharak)


# ########yameen-right-R-3
# directory_path = 'R'
# feature_list_yameen ,label_list_yameen =extract_features_labels(directory_path,3)
# X_yameen = np.array(feature_list_yameen) 
# y_yameen = np.array(label_list_yameen) 

X=np.concatenate((X_yasar, X_qif, X_taharak), axis=0)

y=np.concatenate((y_yasar, y_qif, y_taharak), axis=0)
sumoverallAcuracy=0
SumLAcuracy=0
sumRAcuracy=0
sumQAcuracy=0
sumTAcuracy=0
for iteration in range(1):
# Split data into train and test sets
    # Xt_yasar, Xs_yasar, yt_yasar, ys_yasar = train_test_split(X_yasar, y_yasar, test_size=5)
    # Xt_qif, Xs_qif, yt_qif, ys_qif = train_test_split(X_qif, y_qif, test_size=5)
    # Xt_taharak, Xs_taharak, yt_taharak, ys_taharak = train_test_split(X_taharak, y_taharak, test_size=5)
    # Xt_yameen, Xs_yameen, yt_yameen, ys_yameen = train_test_split(X_yameen, y_yameen, test_size=5)
    Xt_yasar = X_yasar[:10]
    Xs_yasar = X_yasar[10:]
    yt_yasar = y_yasar[:10]
    ys_yasar = y_yasar[10:]
    Xt_qif = X_qif[:10]
    Xs_qif = X_qif[10:]
    yt_qif = y_qif[:10]
    ys_qif = y_qif[10:]
    Xt_taharak = X_taharak[:10]
    Xs_taharak = X_taharak[10:]
    yt_taharak = y_taharak[:10]
    ys_taharak = y_taharak[10:]

    X_train=np.concatenate((Xt_yasar, Xt_qif, Xt_taharak), axis=0)
    X_test=np.concatenate((Xs_yasar, Xs_qif, Xs_taharak), axis=0)
    y_train=np.concatenate((yt_yasar, yt_qif, yt_taharak), axis=0)
    y_test=np.concatenate((ys_yasar, ys_qif, ys_taharak), axis=0)



    # Train a classifier (Random Forest as an example)
    classifier = RandomForestClassifier(n_estimators=1000,random_state=1000)
    classifier.fit(X_train, y_train)

    # Predict on test set
    y_pred = classifier.predict(X_test)
    print(y_pred)
    print(classifier.predict_proba(X_test)[0])