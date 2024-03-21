import os
import pandas as pd
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import pickle
from collections import Counter
from pydub import AudioSegment
from io import BytesIO
import wave
import math
import uuid
import streamlit as st
import joblib 

# Load the LSTM model
def load_lstm_model():
    model_path = "lstm_audio_model (2).joblib"
    model = joblib.load(model_path)
    return model

# Load the Random Forest model
def load_random_forest_model():
    model_path = "myRandomForest (3).pkl"
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    return model

# Function to chop audio into snippets
def chop_new_audio(audio_data, folder):
    os.makedirs(folder, exist_ok=True)
    audio = wave.open(audio_data, 'rb')
    frame_rate = audio.getframerate()
    n_frames = audio.getnframes()
    window_size = 2 * frame_rate
    num_secs = int(math.ceil(n_frames / frame_rate))
    last_number_frames = 0

    for i in range(num_secs):
        shortfilename = str(uuid.uuid4())
        snippetfilename = f"{folder}/{shortfilename}snippet{i+1}.wav"
        snippet = wave.open(snippetfilename, 'wb')
        snippet.setnchannels(2)
        snippet.setsampwidth(audio.getsampwidth())
        snippet.setframerate(frame_rate)
        snippet.setnframes(audio.getnframes())
        snippet.writeframes(audio.readframes(window_size))
        audio.setpos(audio.tell() - 1 * frame_rate)

        if last_number_frames < 1:
            last_number_frames = snippet.getnframes()
        elif snippet.getnframes() != last_number_frames:
            os.rename(snippetfilename, f"{snippetfilename}.bak")
        snippet.close()

# Load models
lstm_model = load_lstm_model()
random_forest_model = load_random_forest_model()

# Function to extract MFCC features and chop audio
def extract_mfcc(audio_file, max_length=100):
    audiofile, sr = librosa.load(audio_file)
    fingerprint = librosa.feature.mfcc(y=audiofile, sr=sr, n_mfcc=20)
    if fingerprint.shape[1] < max_length:
        pad_width = max_length - fingerprint.shape[1]
        fingerprint_padded = np.pad(fingerprint, pad_width=((0, 0), (0, pad_width)), mode='constant')
        return fingerprint_padded.T
    elif fingerprint.shape[1] > max_length:
        return fingerprint[:, :max_length].T
    else:
        return fingerprint.T

# Function to predict on new audio snippets
def predict_cry(audio_file, model):
    audiofile, sr = librosa.load(audio_file)
    mfcc_features = extract_mfcc(audio_file)
    mfcc_features_flat = mfcc_features.reshape(-1)
    if len(mfcc_features_flat) < 2000:
        mfcc_features_flat = np.pad(mfcc_features_flat, (0, 2000 - len(mfcc_features_flat)))
    elif len(mfcc_features_flat) > 2000:
        mfcc_features_flat = mfcc_features_flat[:2000]
    prediction = model.predict([mfcc_features_flat])
    return prediction[0]

# Streamlit app
def app():
    st.title('Baby Cry Classification')

    # Audio upload
    uploaded_file = st.file_uploader("Upload audio file", type=["wav", "mp3"])

    if uploaded_file is not None:
        st.audio(uploaded_file)

        # Save uploaded file
        file_bytes = uploaded_file.read()
        with open("uploaded_audio.wav", "wb") as f:
            f.write(file_bytes)

        # Chop audio
        chop_new_audio(BytesIO(file_bytes), "audio_snippets")

        # Predictions
        lstm_predictions = []
        rf_predictions = []

        for filename in os.listdir("audio_snippets"):
            if filename.endswith(".wav"):
                lstm_prediction = predict_cry(os.path.join("audio_snippets", filename), lstm_model)
                rf_prediction = predict_cry(os.path.join("audio_snippets", filename), random_forest_model)
                lstm_predictions.append(lstm_prediction)
                rf_predictions.append(rf_prediction)

        lstm_counter = Counter(lstm_predictions)
        rf_counter = Counter(rf_predictions)

        st.subheader("LSTM Model Predictions:")
        st.write(lstm_counter)

        st.subheader("Random Forest Model Predictions:")
        st.write(rf_counter)

# Run the app
if __name__ == "__main__":
    app()
