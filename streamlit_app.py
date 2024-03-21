import streamlit as st
import os
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pickle

# Define function to extract MFCC features and chop audio
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

# Define function to load audio data and extract features
def load_data(directory):
    raw_audio = {}
    directories = ['hungry', 'belly_pain', 'burping', 'discomfort', 'tired']
    for directory in directories:
        path = os.path.join(directory, 'Data /Data Source/donateacry_corpus_cleaned_and_updated_data/', directory)
        for filename in os.listdir(path):
            if filename.endswith(".wav"):
                raw_audio[os.path.join(path, filename)] = directory
    
    X, y = [], []
    max_length = 100
    for i, (audio_file, label) in enumerate(raw_audio.items()):
        mfcc_features = extract_mfcc(audio_file, max_length=max_length)
        X.append(mfcc_features)
        y.append(label)

    X = np.array(X)
    y = np.array(y)
    X_flat = X.reshape(X.shape[0], -1)
    y_flat = y

    return X_flat, y_flat

# Function to train and evaluate models
def train_evaluate_models(X_train, y_train, X_test, y_test):
    models = [
        ('Random Forest', RandomForestClassifier(n_estimators=25, max_features=5)),
        ('Logistic Regression', LogisticRegression()),
        ('Decision Tree', DecisionTreeClassifier()),
        ('SVM', SVC()),
    ]

    results = {}
    for model_name, model in models:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        results[model_name] = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall}
    
    return results

# Function to pickle the best model
def pickle_model(model, modelname):
    directory = 'models'
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(os.path.join(directory, str(modelname) + '.pkl'), 'wb') as f:
        pickle.dump(model, f)

# Main function
def main():
    st.title('Audio Classification')

    # Load data
    X, y = load_data('/content/drive/MyDrive/3rd year projects/Thesis/Thesis 1/Data')
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train and evaluate models
    results = train_evaluate_models(X_train, y_train, X_test, y_test)

    # Display results
    st.write("Model Evaluation Results:")
    for model_name, metrics in results.items():
        st.write(f"Model: {model_name}")
        st.write(f"Accuracy: {metrics['Accuracy']}")
        st.write(f"Precision: {metrics['Precision']}")
        st.write(f"Recall: {metrics['Recall']}")

        # Pickle the best model
        if metrics['Accuracy'] == max([metrics['Accuracy'] for metrics in results.values()]):
            best_model = models[model_name]()
            best_model.fit(X_train, y_train)
            pickle_model(best_model, model_name)

if __name__ == '__main__':
    main()
