import numpy as np
import librosa

def extract_mfcc(audio, sr=16000, n_mfcc=13):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    delta_mfcc = librosa.feature.delta(mfcc)
    return np.concatenate([np.mean(mfcc, axis=1), np.mean(delta_mfcc, axis=1)])

def extract_ncc(audio, sr=16000):
    chroma = librosa.feature.chroma_cqt(y=audio, sr=sr)
    return np.mean(chroma, axis=1)

def extract_features(audio, sr=16000):
    # MFCC Features
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13).mean(axis=1)
    # Delta-MFCC
    delta_mfcc = librosa.feature.delta(mfcc)
    # Chromagram
    chroma = librosa.feature.chroma_cqt(y=audio, sr=sr).mean(axis=1)
    # Spectral Centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr).mean()
    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y=audio).mean()

    return np.concatenate([mfcc, delta_mfcc, chroma, [spectral_centroid, zcr]])
