# Security Voice code Access
Task 5 – Security Voice-code Access
Design and implement the software component of Security Voice-code Access. Based on the fingerprint and
spectrogram concepts, the software can be trained on 8 individuals and work in two operation modes:
Mode 1 – Security voice code: where the access is not granted except for a specific pass-code sentence. Any of the
three following sentences is considered a valid passcode: “Open middle door”, “Unlock the gate”, “Grant me access”.
Each group is free to pick any other sentence(s) as long as there are no similar words among the three chosen sentences.

Mode 2 – Security voice fingerprint: where the access is granted to a specific individual(s) who says the valid pass-
code sentence. The software should have settings UI that allow the user to pick which individual of the original 8 users is

granted access. Access can be granted to one or more individuals.
The UI should provide the following elements:
- A button to start recording the voice-code,
- A spectrogram viewer for the spoken voice-code,
- A summary for the analysis results showing 1) a table with how much the spoken sentence matches each of the
saved three passcode sentences, and 2) a table with a table with how much the spoken voice matches each of
the 8 saved individuals,
- Some UI element that indicates the results of the algorithm whether it’s “Access gained” or “Access denied”.
