import os
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
from datasets import load_dataset
from IPython.display import Audio, display
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier


# --------------------------------------------------------------
# Settings
# --------------------------------------------------------------

os.environ["HF_DATASETS_OFFLINE"] = "1"
n_mfcc = 12 #amount of extracted MFCCs
use_extra = True #enables extraction of additional features
bandwidth = 2000 #parameter for spectral bandwitdh


# --------------------------------------------------------------
# EDA utilities
# --------------------------------------------------------------

def plot_confusion_matrix(cm, labels, title):
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))
    plt.imshow(cm_norm, cmap="Blues", interpolation="nearest")
    plt.title(title)
    plt.colorbar()

    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels, rotation=45, ha="right")
    plt.yticks(ticks, labels)

    for i in range(len(labels)):
        for j in range(len(labels)):
            value = cm_norm[i, j]
            color = "white" if value > 0.5 else "black"
            plt.text(j, i, f"{value:.2f}", ha="center", va="center", color=color)

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()


# --------------------------------------------------------------
# Cleaning functions
# --------------------------------------------------------------

def filter_empty(entry):
    arr = entry["audio"]["array"]
    if arr is None or len(arr) == 0:
        return False
    rms = np.sqrt(np.mean(arr**2)) #rms measures the energy in the signal. very little rms-value -> silence
    return rms > 1e-4
    

def filter_unlabeled(entry):
    return entry["genre"] is not None

def clean_dataset(dataset):
    ds = dataset.filter(filter_empty)
    print(f"After removing empty audio: {len(ds)}")
    ds = ds.filter(filter_unlabeled)
    print(f"After removing unlabeled audio: {len(ds)}")
    return ds


# --------------------------------------------------------------
# Feature extraction
# --------------------------------------------------------------

#with mean and maxima we can detect the dynamic of the soundsignal over time

def extract_features(entry, n_mfcc, use_extra, bandwith=None):
    y = entry["audio"]["array"]
    sr = entry["audio"]["sampling_rate"]
    features = []

    features = []
    
    # ZCR
    zcr = librosa.feature.zero_crossing_rate(y)
    features.extend([np.mean(zcr), np.std(zcr), np.max(zcr)]) # NEU: std und max

    # Spectral Centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    features.extend([np.mean(spectral_centroid), np.std(spectral_centroid), np.max(spectral_centroid)]) # NEU: std und max

    # Spectral Rolloff
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features.extend([np.mean(rolloff), np.std(rolloff), np.max(rolloff)]) # NEU: std und max

    # Chroma Frequencies
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features.extend([np.mean(chroma), np.std(chroma), np.max(chroma)]) # NEU: std und max

    # MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    # insert mean (done in baseline) and std and max for every MFCCs 
    for i in range(n_mfcc):
        features.extend([np.mean(mfcc[i]), np.std(mfcc[i]), np.max(mfcc[i])])
    
    # OPTIONAL: add Spectral Bandwidth/RMS Energy , if use_extra is True 
    if use_extra:
        # Spectral Bandwidth
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        features.extend([np.mean(spec_bw), np.std(spec_bw)]) # Hier nur mean/std
        
        # RMS Energy
        rms = librosa.feature.rms(y=y)
        features.extend([np.mean(rms), np.std(rms), np.max(rms)])
        
    return features

def extract_all_features(dataset, n_mfcc=13, use_extra=True):
    X, y = [], []
    for entry in tqdm(dataset, desc="Feature extraction"):
        X.append(extract_features(entry, n_mfcc=n_mfcc, use_extra=use_extra, bandwith=bandwidth))
        y.append(entry["genre"])
    return np.array(X), np.array(y)


# --------------------------------------------------------------
# Modeling functions
# --------------------------------------------------------------

def train_log_reg(X_train, y_train):
    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)
    return model

def train_svm_model(X_train, y_train):
    model = SVC(kernel="rbf", C=10, gamma="scale")
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, labels, title):
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    print(f"{title} accuracy: {acc:.3f}")
    cm = confusion_matrix(y_test, pred)
    plot_confusion_matrix(cm, labels, f"{title} Confusion Matrix")
    return acc


# --------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------

if __name__ == "__main__":

    dataset = load_dataset("unibz-ds-course/audio_assignment", split="train")

    print("Cleaning dataset...")
    cleaned = clean_dataset(dataset)

    print("Extracting features...")
    X, y = extract_all_features(cleaned)

    print("Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    #adding dimensionreduction (PCA)
    print("Applying PCA...")
    pca = PCA(n_components=0.95)
    X_reduced = pca.fit_transform(X_scaled)
    print(f"Features reduced from {X_scaled.shape[1]} to {X_reduced.shape[1]} dimensions")

    print("Splitting data..")
    X_train, X_test, y_train, y_test = train_test_split(
        X_reduced, y, test_size=0.2, random_state=42
    )

    labels = dataset.features["genre"].names

    print("Training Logistic Regression...")
    log_model = train_log_reg(X_train, y_train)
    evaluate_model(log_model, X_test, y_test, labels, "Logistic Regression")

    print("Training SVM...")
    svm_model = train_svm_model(X_train, y_train)
    evaluate_model(svm_model, X_test, y_test, labels, "SVM")
