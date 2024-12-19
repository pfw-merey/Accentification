import argparse
import librosa
import numpy as np
from tensorflow.keras.models import load_model

# Function to extract features from an audio file
def extract_features(audio, sr=16000):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13).mean(axis=1)
    delta_mfcc = librosa.feature.delta(mfcc)
    return np.concatenate([mfcc, delta_mfcc])

def normalize_features(features):
    return (features - np.mean(features)) / np.std(features)

# Function to predict the accent and debug the process
def predict_accent(model_path, audio_file):
    # Load and preprocess the audio file
    audio, sr = librosa.load(audio_file, sr=16000)
    features = extract_features(audio)
    features = normalize_features(features)
    features = features[np.newaxis, ..., np.newaxis]  # Reshape to (1, num_features, 1)

    # Load the trained model
    model = load_model(model_path)

    # Predict the accent
    predictions = model.predict(features)
    print(f"Prediction Probabilities: {predictions}")
    # Log feature statistics
    print(f"Real-world Features: Mean={np.mean(features)}, Std={np.std(features)}")

    if predictions[0][0] > 0.5:  # Adjust threshold
        predicted_accent = "American"
    else:
        predicted_accent = "British"

    return predicted_accent

# Main function for handling command-line arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict the accent of an audio file using a trained model.")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained model file (e.g., v1.keras).")
    parser.add_argument("--file", type=str, required=True, help="Path to the audio file to be classified (e.g., audio.wav).")

    args = parser.parse_args()

    # Predict the accent
    predicted_accent = predict_accent(args.model, args.file)
    print(f"Predicted Accent: {predicted_accent}")
