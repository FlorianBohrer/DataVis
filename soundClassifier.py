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


# --------------------------------------------------------------
# Settings
# --------------------------------------------------------------

os.environ["HF_DATASETS_OFFLINE"] = "1"


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
    if np.all(arr == 0):
        return False
    return True
    

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

def extract_features(entry):
    arr = entry["audio"]["array"]
    sr = entry["audio"]["sampling_rate"]

    mfcc = librosa.feature.mfcc(y=arr, sr=sr, n_mfcc=13)
    rms = librosa.feature.rms(y=arr)[0]
    zcr = librosa.feature.zero_crossing_rate(arr)[0]
    contrast = librosa.feature.spectral_contrast(y=arr, sr=sr)

    return np.concatenate([
        mfcc.mean(axis=1), mfcc.var(axis=1),
        contrast.mean(axis=1), contrast.var(axis=1),
        [np.mean(rms), np.var(rms), np.mean(zcr), np.var(zcr)]
    ])

def extract_all_features(dataset):
    X, y = [], []
    for entry in tqdm(dataset, total=len(dataset)):
        X.append(extract_features(entry))
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

    print("Splitting data..")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    labels = dataset.features["genre"].names

    print("Training Logistic Regression...")
    log_model = train_log_reg(X_train, y_train)
    evaluate_model(log_model, X_test, y_test, labels, "Logistic Regression")

    print("Training SVM...")
    svm_model = train_svm_model(X_train, y_train)
    evaluate_model(svm_model, X_test, y_test, labels, "SVM")
