import numpy as np
import torch
torch.set_num_threads(1)
import pyaudio
import queue
import threading

import os
import queue
import json
import sounddevice as sd
from vosk import Model, KaldiRecognizer

import random
import argparse
import os
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

#------------ Training Intent Classification ---------------#

examples_dict = {
        "sit": ["sit", "sit down", "take a seat", "please sit down", "sit still"],
        "come": ["come", "come here", "follow me", "approach me", "get over here"],
        "fetch": ["fetch", "fetch the ball", "bring it here", "go get it", "retrieve it"],
        "spin": ["spin", "turn around", "do a spin", "make a circle"],
        "stay": ["stay", "stay there", "don’t move", "hold still", "freeze"]
    }

# FLATTEN DATA

data = []
for command, examples in examples_dict.items():
    for example in examples:
        data.append((example, command))
random.shuffle(data)

print("Training traditional Term Frequency-Inverse Document Frequency + Logistic Regression classifier...")

texts = [x[0] for x in data]
labels = [x[1] for x in data]

X_train, X_val, y_train, y_val = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

# TRAIN

clf = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("lr", LogisticRegression(max_iter=1000))
])
clf.fit(X_train, y_train)
y_pred = clf.predict(X_val)
print("Accuracy:", accuracy_score(y_val, y_pred))


#------------------------ Silero VAD Setup-----------------------------#
""" vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=True) """

vad_model, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    force_reload=False,  # use cached version, no network needed
    trust_repo=True
)

(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

# Taken from utils_vad.py
def validate(model,
             inputs: torch.Tensor):
    with torch.no_grad():
        outs = model(inputs)
    return outs

# Provided by Alexander Veysov
def int2float(sound):
    abs_max = np.abs(sound).max()
    sound = sound.astype('float32')
    if abs_max > 0:
        sound *= 1/32768
    sound = sound.squeeze()  # depends on the use case
    return sound

vad_iterator = VADIterator(
    vad_model,
    threshold=0.5,
    sampling_rate=16000,
    min_silence_duration_ms=500,  # how long silence before "speech ended"
    speech_pad_ms=100             # pad speech start/end by 100ms
)



# ------ Vosk Setup ------#
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "speech_to_text/models", "vosk-model-small-en-us-0.15")

vosk_model = Model(model_path)
recognizer = KaldiRecognizer(vosk_model, 16000)



# ----------------- Real time run -------------------#
q = queue.Queue()
speech_buffer = []  # accumulate speech chunks
isSpeaking = False

def callback(indata, frames, time, status):
    if status:
        print(status, flush=True)
    q.put(bytes(indata))

with sd.RawInputStream(samplerate=16000, blocksize=512, dtype="int16",
                       channels=1, callback=callback):
    print("Listening for audio...")

    while True:
        try:
            data = q.get()
            
            audio_int16 = np.frombuffer(data, np.int16)
            audio_float32 = int2float(audio_int16)
            audio_tensor = torch.frombuffer(audio_float32, dtype=torch.float32)
            result = vad_iterator(audio_tensor, return_seconds=True)

            if result:
                if 'start' in result:
                    print(f"Speech started: {result['start']}s")
                    isSpeaking = True
                    speech_buffer = []

                if 'end' in result:
                    print(f"Speech ended: {result['end']}s")
                    # Feed accumulated buffer to Vosk
                    full_audio = b"".join(speech_buffer)

                    key = ""
                    if recognizer.AcceptWaveform(full_audio):
                        result = recognizer.Result()
                        key = "text"
                    else:
                        result = recognizer.PartialResult()
                        key = "partial"

                    text = json.loads(result)[key]            
                    if text:
                        print(f"{key}: {text}")
                        # intent classification
                        # INFERENCE
                        pred = clf.predict([text])[0]
                        print(f"{text} --> {pred}")



                    recognizer.Reset()
                    vad_iterator.reset_states()
            
            if isSpeaking:
                speech_buffer.append(data)
        except KeyboardInterrupt:
            print("Stopping recording...")
            break




