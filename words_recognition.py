import librosa
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os
import librosa
import numpy as np


###Made By: Nour Aldeen Hassan Khalaf
###And Youssef Mohamed Abdelnaby Darwish
#######funcion for loading the voice recordes
def extract_features_labels(file_path,labelnum):
    feature = []
    label = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".wav"):  # Assuming audio files are in WAV format
            file_path = os.path.join(directory_path, filename)
            
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
    return feature,label

######## yasar-Left-L-0
directory_path = 'L'
feature_list_yasar ,label_list_yasar =extract_features_labels(directory_path,0)
X_yasar = np.array(feature_list_yasar) 
y_yasar = np.array(label_list_yasar) 

######qif_q-1

directory_path = 'q'
feature_list_qif ,label_list_qif =extract_features_labels(directory_path,1)
X_qif = np.array(feature_list_qif)
y_qif = np.array(label_list_qif) 
 

#########taharak-t-2

directory_path = 't'
feature_list_taharak ,label_list_taharak =extract_features_labels(directory_path,2)
X_taharak = np.array(feature_list_taharak) 
y_taharak = np.array(label_list_taharak) 


########yameen-right-R-3
directory_path = 'R'
feature_list_yameen ,label_list_yameen =extract_features_labels(directory_path,3)
X_yameen = np.array(feature_list_yameen) 
y_yameen = np.array(label_list_yameen) 

###Made By: Nour Aldeen Hassan Khalaf
###And Youssef Mohamed Abdelnaby Darwish

####combining vectors ahhhhhh akt4ft en el mafrod wa7d
X=np.concatenate((X_yasar, X_qif, X_taharak, X_yameen), axis=0)

y=np.concatenate((y_yasar, y_qif, y_taharak, y_yameen), axis=0)
sumoverallAcuracy=0
SumLAcuracy=0
sumRAcuracy=0
sumQAcuracy=0
sumTAcuracy=0
for iteration in range(10):
# Split data into train and test sets
    Xt_yasar, Xs_yasar, yt_yasar, ys_yasar = train_test_split(X_yasar, y_yasar, test_size=0.33)
    Xt_qif, Xs_qif, yt_qif, ys_qif = train_test_split(X_qif, y_qif, test_size=0.33)
    Xt_taharak, Xs_taharak, yt_taharak, ys_taharak = train_test_split(X_taharak, y_taharak, test_size=0.33)
    Xt_yameen, Xs_yameen, yt_yameen, ys_yameen = train_test_split(X_yameen, y_yameen, test_size=0.33)

    X_train=np.concatenate((Xt_yasar, Xt_qif, Xt_taharak, Xt_yameen), axis=0)
    X_test=np.concatenate((Xs_yasar, Xs_qif, Xs_taharak, Xs_yameen), axis=0)
    y_train=np.concatenate((yt_yasar, yt_qif, yt_taharak, yt_yameen), axis=0)
    y_test=np.concatenate((ys_yasar, ys_qif, ys_taharak, ys_yameen), axis=0)



    # Train a classifier (Random Forest as an example)
    classifier = RandomForestClassifier(n_estimators=1000, random_state=42)
    classifier.fit(X_train, y_train)

    # Predict on test set
    y_pred = classifier.predict(X_test)

    # Evaluate accuracy
    ####each word accuracy
    per_sample_accuracy = y_test == y_pred
    t_sum=0
    l_sum=0
    r_sum=0
    q_sum=0
    ##loop for sum of truely predicted value for each
    for pre,test in  zip(y_pred,y_test):
        
        if(pre==test):
            x=1
        else:
            x=0
        
        if pre==0:
            l_sum+=x
        if pre==1:
            q_sum+=x
        if pre==2:
            t_sum+=x
        if pre==3:
            r_sum+=x

    #accuracy for each
    t_per=t_sum/(y_test.size/4)
    l_per=l_sum/(y_test.size/4)
    r_per=r_sum/(y_test.size/4)
    q_per=q_sum/(y_test.size/4)

    ###ovar all accuracy
    accuracy = accuracy_score(y_test, y_pred)
    #######adding current results for total
    sumoverallAcuracy+=accuracy
    SumLAcuracy+=l_per
    sumRAcuracy+=r_per
    sumQAcuracy+=q_per
    sumTAcuracy+=t_per
    ###printing all details for current iteration
    print("iteration = ",iteration+1)
    print("Number of features/sample == ",len(feature_list_qif))
    print("Number of training samples for each word = ",y_train.size/4)
    print("Number of testing samples for each word = ",y_test.size/4)
    print("Average Accuracy for each word : ")
    print("yasar = ",l_per*100,"%")
    print("yameen = ",r_per*100,"%")
    print("qif = ",q_per*100,"%")
    print("taharak = ",t_per*100,"%")
    print("OverAll Accuracy = ", accuracy*100,"%")
    print("------------------next iteration----------------------")

print("-----------------------------------------------------------")
print("Final Results")
print("Average Accuracy for each word : ")
print("yasar = ",(SumLAcuracy/10)*100,"%")
print("yameen = ",(sumRAcuracy/10)*100,"%")
print("qif = ",(sumQAcuracy/10)*100,"%")
print("taharak = ",(sumTAcuracy/10)*100,"%")
print("------------------------------------------")
print("OverAll Accuracy = ", (sumoverallAcuracy/10)*100,"%")
print("------------------------------------------")
print("Made By: Nour Aldeen Hassan Khalaf ")
print("And Youssef Mohamed Abdelnaby Darwish ")
###Made By: Nour Aldeen Hassan Khalaf
###And Youssef Mohamed Abdelnaby Darwish