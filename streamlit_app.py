# Import necessary libraries
from sklearn import model_selection
import streamlit as st
import librosa
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import pickle

# Define the Streamlit app
def app():
    st.title('Infant Cry Classification')

    # Display choice of classifier
    options = ['LSTM', 'Random Forest']
    selected_option = st.selectbox('Select the classifier', options)

    # Define model loading functions based on classifier type
    def load_lstm_model():
        try:
            model_path = "lstm_audio_model (2).joblib"
            model = joblib.load(model_path)
            return model
        except FileNotFoundError:
            st.error(f"LSTM model not found at '{model_path}'. Please ensure the model exists.")
            return None

    def load_random_forest_model():
        try:
            model_path = "myRandomForest (3).pkl"
            with open(model_path, "rb") as file:
                model = pickle.load(file)
            return model
        except FileNotFoundError:
            st.error(f"Random Forest model not found at '{model_path}'. Please ensure the model exists.")
            return None

    if selected_option == 'Random Forest':
        model = load_random_forest_model()
    else:
        model = load_lstm_model()
        if model is None:
            st.warning("Model loading failed. Classification functionality unavailable.")

    # Define function to predict cry
    def predict_cry(audio_file):
        try:
            # Preprocess audio (extract MFCC features)
            audio, sr = librosa.load(audio_file)
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=1)
            mfcc_scaled = StandardScaler().fit_transform(mfcc.T)
            mfcc_df = pd.DataFrame(mfcc_scaled.T)

            prediction = model.predict(mfcc_df)

            # Get the class label
            class_names = model.classes_
            predicted_class = class_names[prediction[0]]
            return predicted_class
        except Exception as e:
            st.error("Error occurred during prediction.")
            return None

    # Audio upload
    uploaded_file = st.file_uploader("Upload audio file", type=["wav"])

    if uploaded_file is not None:
        # Perform classification if audio file is uploaded
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())

        predicted_cry = predict_cry(uploaded_file.name)
        if predicted_cry is not None:
            st.success(f"Predicted cry: {predicted_cry}")
# Run the app
if __name__ == "__main__":
    app()
