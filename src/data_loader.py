import random
import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def oversample_data(features, labels, target_count, target_label):
    """Oversample the minority class."""
    minority_features = [f for f, l in zip(features, labels) if l == target_label]
    additional_samples = random.choices(minority_features, k=target_count - len(minority_features))
    oversampled_features = features + additional_samples
    oversampled_labels = labels + [target_label] * len(additional_samples)
    return oversampled_features, oversampled_labels


def augment_audio(audio, sr=16000):
    # Add slight noise
    noise = np.random.normal(0, 0.002, audio.shape)
    audio_with_noise = audio + noise
    # Slight pitch shift
    audio_pitched = librosa.effects.pitch_shift(audio, sr=sr, n_steps=1)
    return [audio, audio_with_noise, audio_pitched]


def load_data(data_path):
    label_mapping = {"american": 0, "british": 1}
    features, labels = [], []

    for label in os.listdir(data_path):
        label_path = os.path.join(data_path, label)
        if os.path.isdir(label_path):
            for file in os.listdir(label_path):
                file_path = os.path.join(label_path, file)
                try:
                    audio, sr = librosa.load(file_path, sr=16000)
                    features.append(audio)
                    labels.append(label_mapping[label])
                except Exception as e:
                    print(f"Error loading file {file_path}: {e}")

    # Oversample the British class to match USA sample count (373)
    features, labels = oversample_data(features, labels, target_count=373, target_label=1)

    return features, labels

