import os

exists = True
i = 1
while exists:
    if os.path.exists(f"recording{i}.wav"):
        os.remove(f"recording{i}.wav")
        i += 1
    else:
        exists = False