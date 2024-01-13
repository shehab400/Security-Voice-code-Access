# Security Voice-code Access Application

## Overview

The Security Voice-code Access application is a security system designed to provide access based on voice recognition. The software component utilizes fingerprint and spectrogram concepts, incorporating machine learning scikit-learn classifiers (Random Forest) for enhanced security. The application operates in two modes: Security voice code and Security voice fingerprint.

## Operation Modes

### Mode 1 - Security Voice Code

In this mode, access is granted only when a specific pass-code sentence is spoken. The valid passcode sentences are:
1. "Open middle door"
2. "Unlock My gate "
3. "Grant me access"

Users can choose alternative sentences, ensuring no similarity among the selected phrases.

### Mode 2 - Security Voice Fingerprint

Access is granted based on the voice fingerprint of specific individuals who say the valid passcode sentence. The application allows users to select which individual(s) from the original 8 users are granted access. Access can be granted to one or more individuals.

## Machine Learning Component

The application incorporates a Random Forest classifier for voice recognition. The classifier is trained on the voice features of the 8 individuals during the training phase and used to predict the speaker during the testing phase.

## User Interface (UI) Features

The UI provides the following elements:

1. **Record Voice-code Button:**
   - Initiates the voice-code recording process.

2. **Spectrogram Viewer:**
   - Displays the spectrogram of the spoken voice, providing a visual representation.

3. **Analysis Summary:**
   - Shows analysis results, including:
      - A table indicating how much the spoken sentence matches each of the saved three passcode sentences.
      - A table illustrating the similarity of the spoken voice to each of the 8 saved individuals.

4. **Algorithm Results Indicator:**
   - UI element indicating whether the algorithm results in "Access Gained" or "Access Denied."

## Getting Started

To use the Security Voice-code Access application:

1. Clone the repository to your local machine.
2. Install the required dependencies mentioned in the documentation.
3. Run the application and use the provided UI elements for voice-code recording and analysis.




For developers interested in contributing or extending the functionality:

- The application is built using [Python].
- Follow the coding standards and contribute to the enhancement of features.
- Create a pull request to submit changes.


## Acknowledgments

### Contributors

- Abdulrahman Hesham
- Shehap Elhadary
- Mohamed Ibrahim

