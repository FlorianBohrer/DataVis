# Music Genre Classification from Audio

A classical machine-learning pipeline that classifies 30-second music clips into one of ten
genres (`blues`, `classical`, `country`, `disco`, `hiphop`, `jazz`, `metal`, `pop`, `reggae`,
`rock`) directly from the raw audio signal.

The dataset is intentionally noisy: it contains empty (silent) clips and unlabeled samples.
The workflow below covers the full pipeline:

1. **Exploratory data analysis** &ndash; class balance and audio-duration distribution
2. **Data cleaning** &ndash; removing silent and unlabeled samples
3. **Feature engineering** &ndash; MFCCs (+ deltas), spectral, rhythmic and bass features
4. **Modeling** &ndash; comparing several sklearn classifiers, hyperparameter tuning,
   cross-validation and per-class evaluation

The audio data is loaded via the Hugging&nbsp;Face `datasets` library.
## Setup
Let's install all required dependencies:

- **datasets**: Access to large-scale datasets.
- **librosa**: Tools for audio analysis.
- **pandas** & **numpy**: Tabular data manipulation and numerical operations.
- **scikit-learn**: Machine learning algorithms and tools.
- **tqdm**: Progress bar.

You might be familiar with most of these already.
%%capture
!pip install datasets==3.5.0 librosa pandas numpy scikit-learn tqdm
And import the necessary modules
import os
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
from datasets import load_dataset
from IPython.display import Audio, display
from tqdm import tqdm
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier


from sklearn.metrics import f1_score
### Dataset Description

The dataset consists of music samples from various genres:

- **Genres**: `Blues`, `Classical`, `Country`, `Disco`, `HipHop`, `Jazz`, `Metal`, `Pop`, `Reggae`, and `Rock`.

It is intentionally messy and includes some **unlabeled data** and **empty audio files**, which are handled in the data-cleaning step below.
Let's download the audio dataset using the Hugging Face datasets library.
dataset = load_dataset("unibz-ds-course/audio_assignment", split="train")
dataset
print(f"Num of samples in the dataset: {len(dataset)}")
## Exploratory Data Analysis

Let's start by inspecting a single sample to understand the structure of each entry &ndash;
the raw waveform, its sampling rate, duration and genre label.
entry = dataset[150]

arr = entry['audio']['array']
sr = entry['audio']['sampling_rate']

print(f"Element: {entry}")
print(f"File Path: {entry['file']}")
print(f"Number of Samples: {len(arr)}")
print(f"Sampling Rate: {sr} Hz")

audio_length_seconds = len(arr) / sr
print(f"Audio Length: {audio_length_seconds:.2f} seconds")

genre_id = entry['genre']
genre_label = dataset.features['genre'].int2str(genre_id)
print(f"Genre (ID): {genre_id}")
print(f"Genre (Label): {genre_label}")

display(Audio(arr, rate=sr))
### Class Distribution

A bar chart of samples per genre shows whether the dataset is balanced or whether some
classes are underrepresented &ndash; important context for choosing evaluation metrics later.
df = pd.DataFrame(dataset)
genre_counts = df['genre'].value_counts()
genre_counts.index = genre_counts.index.map(lambda x: dataset.features['genre'].int2str(int(x)))
genre_counts = genre_counts.sort_values(ascending=False)

# create bar chart
plt.figure(figsize=(12, 6))
bars = plt.bar(genre_counts.index, genre_counts.values, color='skyblue')

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 5, int(yval), ha='center', va='bottom', fontsize=10)

plt.title('Distribution of Music Genres (Classes) in the Dataset')
plt.xlabel('Music Genre')
plt.ylabel('Number of Samples')
plt.xticks(rotation=45, ha='right')
plt.ylim(0, genre_counts.max() * 1.1)
plt.grid(axis='y', linestyle='--')
plt.tight_layout()

plt.savefig('genre_distribution.png')
plt.show()

### Audio Duration Distribution

The distribution of clip lengths reveals how consistent the recordings are and whether
duration normalization is needed before feature extraction.
def plot_audio_distribution(dataset, title='Distribution of Audio Durations'):
    durations = [len(x['audio']['array']) / x['audio']['sampling_rate'] for x in dataset]

    plt.figure(figsize=(10, 6))
    sns.histplot(durations, bins=30, kde=True, color='orange')
    plt.title(title)
    plt.xlabel('Duration (seconds)')
    plt.ylabel('Count')
    plt.grid(axis='y', alpha=0.3)
    plt.show()

plot_audio_distribution(dataset)
## Data Cleaning

### Removing Empty Audio Samples

Some clips contain only silence (a zero or near-zero signal). These carry no information
and are removed by checking the maximum absolute amplitude against a small threshold.
def filter_empty_samples(entry):
    arr = entry["audio"]["array"]

    if arr is None or len(arr) == 0:
        return False
    if np.max(np.abs(arr))==0:
        return False
    
    return np.max(np.abs(arr)) > 1e-4
filtered_dataset = dataset.filter(filter_empty_samples)
print(f"remaining Samples: {len(filtered_dataset)}")
assert len(filtered_dataset) == 970, "Expected 970 samples after removing empty audio"
### Removing Unlabeled Samples

A number of entries have no genre label. Since this is a supervised task, they are dropped.
def filter_unlabeled_samples(entry):
    return entry["genre"] is not None


filtered_dataset = filtered_dataset.filter(filter_unlabeled_samples)
print(f"remaining Samples: {len(filtered_dataset)}")
assert len(filtered_dataset) == 848, "Expected 848 samples after removing unlabeled data"
## Feature Engineering

With a clean dataset, we can now turn each raw waveform into a fixed-length numerical
feature vector suitable for classical ML models.
### **Mel Frequency Cepstral Coefficients**

**[Mel Frequency Cepstral Coefficients (MFCCs)](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum)** are commonly used in audio analysis to capture key features of sound. They help represent the important characteristics of an audio signal, making them ideal for tasks like music genre classification and speech recognition.

We're not going to dive deep into the complex details of audio processing, but it's useful to know that MFCCs help simplify raw audio data while retaining important information.

#### Basic Steps in MFCC Extraction:
1. **Frequency Domain Conversion**: The audio signal is split into short frames, and we apply the Fourier Transform to convert them from the time domain to the frequency domain.
2. **Mel Scale Mapping**: The frequency spectrum is converted to the Mel scale, which better represents how humans perceive sound, emphasizing lower frequencies.
3. **Logarithm and DCT**: After mapping to the Mel scale, we apply a logarithm and the Discrete Cosine Transform (DCT) to get the MFCCs. These summarize the "cepstral" information of the audio signal.

The parameter `n_mfcc` controls **how many MFCC coefficients** are extracted for each frame. For example, setting `n_mfcc=8` means we extract 8 coefficients, where lower coefficients capture broad audio features, and higher coefficients capture the more finer details.

#### Why MFCCs Are Important:
MFCCs help capture the **tonal quality** of the sound and reduce the complexity of the raw audio signal. By summarizing the audio into a smaller set of features, they allow machine learning models to classify and recognize different types of sounds more effectively.

In this notebook, we'll use the **mean** and **variance** of the MFCCs over time to create a robust feature set for our classification model. Adjusting the `n_mfcc` parameter allows us to control the number of features extracted for each audio sample.

#### **Additional Features**
Consider exploring additional audio features to enhance your model's performance. There are various acoustic properties you could extract from the audio signals, such as zero crossings, harmonic-percussive separation, tempo, spectral centroids, spectral rolloff, chromagram, RMS energy, spectral bandwidth, etc. When working with these features, it's often useful to compute summary statistics like the mean and variance across the audio sample. These summary statistics can capture the overall characteristics and variability of the feature, reducing the dimensionality of your data while retaining important information. Experimenting with these features and their statistical summaries could potentially improve your model's accuracy and robustness in distinguishing between different audio characteristics.

#### **Feature Analysis**
Don´t forget to optimize the use of features, identifying and handling irrelevant and reduntant features. Then use feature ranking to identify which features are more influential, and evaluate quantitatively how many top-features to retain.
def extract_mfcc_features(dataset, n_mfcc):
    mfcc_features = []

    # here we might have used Dataset.map method, unfortunately, it consumes extra memory and runs out of RAM in colab
    for entry in tqdm(dataset, desc="Extracting MFCC Features"):
        audio_array = entry['audio']['array']
        sampling_rate = entry['audio']['sampling_rate']

        mfcc = librosa.feature.mfcc(y=audio_array, sr=sampling_rate, n_mfcc=n_mfcc)

        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_var = np.var(mfcc, axis=1)

        feature_dict = {}

        for i in range(n_mfcc):
            feature_dict[f'mfcc_mean{i+1}'] = mfcc_mean[i]

        for i in range(n_mfcc):
            feature_dict[f'mfcc_var{i+1}'] = mfcc_var[i]

        feature_dict['genre'] = entry['genre']

        mfcc_features.append(feature_dict)

    return mfcc_features
Let's take a look at the output of the function. We will pass there just 2 samples from the dataset.
extract_mfcc_features(dataset.select(range(2)), n_mfcc=5)
The function generates `n_mfcc * 2` features per sample (mean and variance of each
coefficient). MFCCs are effective for audio, but performance can be improved by adding
complementary descriptors such as RMS energy and spectral contrast, explored below.
### RMS & Spectral Contrast

**RMS energy** captures the loudness/intensity of a signal over time &ndash; useful for
separating energetic genres (e.g. metal) from quieter ones (e.g. classical).

**Spectral contrast** measures the difference between peaks and valleys in the spectrum,
which helps distinguish harmonic content and timbre across genres. **Spectral centroid**
is added as a proxy for perceived brightness.
def extract_spectral_contrast_and_centroid(entry):
    audio_array = entry['audio']['array']
    sampling_rate = entry['audio']['sampling_rate']

    feature_dict = {}

    # Spectral Centroid 
    cent = librosa.feature.spectral_centroid(y=audio_array, sr=sampling_rate)[0]
    feature_dict['cent_mean'] = np.mean(cent)
    feature_dict['cent_std'] = np.std(cent)

    # Spectral Contrast
    contrast = librosa.feature.spectral_contrast(y=audio_array, sr=sampling_rate)
    
    contrast_mean = np.mean(contrast, axis=1)
    
    for i in range(contrast.shape[0]):
        feature_dict[f'contrast_mean{i+1}'] = contrast_mean[i]

    return feature_dict
def extract_rms(entry, eps=1e-4):
    arr = entry["audio"]["array"]
    if arr is None or arr.size == 0:
        return 0.0
    rms = np.sqrt(np.mean(arr ** 2))
    return rms



### Additional Features

To better capture genre characteristics, the feature set is extended with MFCC deltas
(first and second order), further spectral statistics (flatness, rolloff, zero-crossing
rate), rhythmic descriptors (tempo, onset strength) and a low-frequency *bass ratio*.
All clips are normalized to a fixed 30-second length so every sample yields a vector of
identical dimension.
def extract_features(entry, n_mfcc=20, target_duration=30.0):
    import numpy as np
    import librosa

    arr = entry["audio"]["array"]
    sr = entry["audio"]["sampling_rate"]

    if arr is None or len(arr) == 0:
        return None

    #Time normalization 
    target_len = int(target_duration * sr)
    if len(arr) > target_len:
        arr = arr[:target_len]
    else:
        arr = np.pad(arr, (0, target_len - len(arr)))

    # MFCCs 
    mfcc = librosa.feature.mfcc(y=arr, sr=sr, n_mfcc=n_mfcc)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    mfcc_features = np.concatenate([
        mfcc.mean(axis=1),
        mfcc.var(axis=1),
        mfcc_delta.mean(axis=1),
        mfcc_delta.var(axis=1),
        mfcc_delta2.mean(axis=1),
        mfcc_delta2.var(axis=1)
    ])

    # Spectral features
    centroid = librosa.feature.spectral_centroid(y=arr, sr=sr)[0]
    flatness = librosa.feature.spectral_flatness(y=arr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=arr, sr=sr)[0]
    zcr = librosa.feature.zero_crossing_rate(arr)[0]
    rms = librosa.feature.rms(y=arr)[0]

    flatness_iqr = np.percentile(flatness, 75) - np.percentile(flatness, 25)

    spectral_features = np.array([
        centroid.mean(), centroid.var(),
        flatness.mean(), flatness.var(),
        rolloff.mean(), rolloff.var(),
        zcr.mean(), zcr.var(),
        rms.mean(), rms.var()
    ])

    #Spectral contrast 
    contrast = librosa.feature.spectral_contrast(y=arr, sr=sr)
    contrast_features = np.concatenate([
        contrast.mean(axis=1),
        contrast.var(axis=1)
    ])

    #Rhythm features (boosted) 
    tempo, _ = librosa.beat.beat_track(y=arr, sr=sr)
    tempo = float(tempo)

    onset_env = librosa.onset.onset_strength(y=arr, sr=sr)

    rhythm_features = np.array([
        tempo,
        tempo,  # intentional duplication
        onset_env.mean(),
        onset_env.var(),
        onset_env.var()  # intentional duplication
    ])

    # Bass dominance (reggae-focused)
    S = np.abs(librosa.stft(arr))
    freqs = librosa.fft_frequencies(sr=sr)

    low_energy = S[freqs < 150].mean()
    mid_energy = S[(freqs >= 150) & (freqs < 800)].mean()

    bass_ratio = float(low_energy / (mid_energy + 1e-6))
    bass_features = np.array([bass_ratio])

    # Final feature vector 
    features = np.concatenate([
        mfcc_features,
        spectral_features,
        contrast_features,
        rhythm_features,
        bass_features
    ])

    return features

def extract_all_features(dataset, n_mfcc=13):
    X, y = [], []
    # Dynamic feature names for plotting
    feature_names = ['zcr_mean', 'zcr_std', 'zcr_max', 'cent_mean', 'cent_std', 'cent_max', 
                     'roll_mean', 'roll_std', 'roll_max', 'chroma_mean', 'chroma_std', 'chroma_max']
    for i in range(n_mfcc):
        feature_names.extend([f'mfcc_{i}_mean', f'mfcc_{i}_std', f'mfcc_{i}_max'])

    for entry in tqdm(dataset, desc="Extracting Audio Features"):
        feat = extract_features(entry, n_mfcc=n_mfcc,)
        if feat is not None:
            X.append(feat)
            y.append(entry["genre"])
            
    return np.array(X), np.array(y), feature_names

### Feature Correlation

Plotting the correlation matrix highlights redundant features. Highly correlated
descriptors carry overlapping information and can be reduced to improve generalization.
def analyze_and_reduce_features(X, feature_names, threshold=0.95):
   

    # Create DataFrame for easier handling
    df = pd.DataFrame(X, columns=feature_names)

    # Compute correlation matrix
    corr = df.corr()

    # Plot correlation heatmap
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8}
    )
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.show()

    # Identify highly correlated features
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [
        column for column in upper.columns
        if any(upper[column].abs() > threshold)
    ]

    # Drop correlated features
    df_reduced = df.drop(columns=to_drop)

    print(f"Removed {len(to_drop)} correlated features")
    print(f"Remaining features: {df_reduced.shape[1]}")

    return df_reduced.values, df_reduced.columns.tolist()


### Feature Importance

Ranking features by a Random Forest's importance scores shows which descriptors contribute
most to the predictions.



def plot_feature_importance(X, y, names):
    #scaling the data for a fair ranking
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    #train random forest for extracting the feature importance
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    rf.fit(X_scaled, y)
    
    #extract the feature importance
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1] 

    # plot
    plt.figure(figsize=(14, 8))
    plt.title("Feature Importance Ranking (Random Forest)")
    plt.bar(range(X.shape[1]), importances[indices], color='darkgreen', align="center")
    plt.xticks(range(X.shape[1]), [names[i] for i in indices], rotation=90)
    plt.xlabel("Feature Name")
    plt.ylabel("Importance Score")
    plt.tight_layout()
    plt.show()
    
    
    print("Top 10 most important Features:")
    for i in range(10):
        print(f"{i+1}. {names[indices[i]]} ({importances[indices[i]]:.4f})")

plot_feature_importance(X, y, feature_names)

## Modeling

### Baseline Pipeline
Let's import all necessary functions and classes from sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
RANDOM_STATE = 42
Function to extract features dataset from initial audio dataset
def prepare_dataset(dataset, n_mfcc=13): #mfcc=20
    X_list = []
    y_list = []

    for entry in tqdm(dataset, desc="Extracting Audio Features"):
        feat = extract_features(entry, n_mfcc=n_mfcc)
        if feat is not None:
            X_list.append(feat)
            y_list.append(entry["genre"])

    X = np.array(X_list)
    y = np.array(y_list)

    # --- feature names ---
    feature_names = []

    # MFCC base + deltas
    for i in range(n_mfcc):
        feature_names.append(f"mfcc_mean_{i}")
    for i in range(n_mfcc):
        feature_names.append(f"mfcc_var_{i}")
    for i in range(n_mfcc):
        feature_names.append(f"mfcc_delta_mean_{i}")
    for i in range(n_mfcc):
        feature_names.append(f"mfcc_delta_var_{i}")
    for i in range(n_mfcc):
        feature_names.append(f"mfcc_delta2_mean_{i}")
    for i in range(n_mfcc):
        feature_names.append(f"mfcc_delta2_var_{i}")

    # Spectral statistics
    feature_names += [
        "centroid_mean", "centroid_var",
        "flatness_mean", "flatness_var",
        "rolloff_mean", "rolloff_var",
        "zcr_mean", "zcr_var",
        "rms_mean", "rms_var"
    ]

    # Spectral contrast (librosa default = 7 bands)
    n_contrast = 7
    for i in range(n_contrast):
        feature_names.append(f"contrast_mean_{i}")
    for i in range(n_contrast):
        feature_names.append(f"contrast_var_{i}")

    # Rhythm + bass
# Rhythm + bass (matches extract_features)
    feature_names += [
        "tempo",
        "tempo_dup",
        "onset_mean",
        "onset_var",
        "onset_var_dup",
        "bass_ratio"
    ]

    # --- safety check ---
    assert X.shape[1] == len(feature_names), \
        f"Feature mismatch: X has {X.shape[1]}, names have {len(feature_names)}"

    print("Feature shape:", X.shape)

    return X, y, feature_names

Now let's prepare train and test data
X, y, feature_names = prepare_dataset(filtered_dataset, n_mfcc=20)

df_feat = pd.DataFrame(X, columns=feature_names)

corr = df_feat.corr()

plt.figure(figsize=(12,10))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(
    corr,
    mask=mask,
    cmap="coolwarm",
    center=0,
    linewidths=0.3,
    cbar_kws={"shrink": 0.8}
)
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.show()



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#### Discussion: Feature Correlation Matrix

The matrix clearly shows that many MFCC-based features are strongly correlated with each
other, meaning they contain overlapping information. In contrast, rhythm and spectral
features such as tempo or spectral contrast are less correlated and therefore add
complementary information. This indicates that feature reduction or regularization is
useful to reduce redundancy and improve model generalization.
The best practice is to use a pipeline because it allows us to streamline preprocessing steps (like scaling) and model training into a single workflow. This ensures that all steps are applied consistently during both training and testing, preventing data leakage and simplifying cross-validation and hyperparameter tuning.

pipeline = Pipeline([
    # Feature scaling is mandatory for audio features
    ('scaler', StandardScaler()),

    # Multiclass logistic regression with regularization
    ('classifier', LogisticRegression(
        max_iter=3000,
        solver="lbfgs",
        multi_class="multinomial",
        class_weight="balanced",
        random_state=42
    ))
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")
### Model Comparison

Several classical classifiers (Logistic Regression, Random Forest, SVM, Gradient Boosting,
k-NN) are trained on the same features and compared on accuracy and weighted F1-score.

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier


models = [
    ('Logistic Regression', LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
    ('Random Forest', RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)),
    ('SVM (RBF)', SVC(kernel='rbf',C=5,gamma=0.01,class_weight='balanced',probability=True,random_state=RANDOM_STATE)),    
    ('Gradient Boosting', GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE)),
    ('k-Nearest Neighbors', KNeighborsClassifier(n_neighbors=5))
]

results = []

print("--- Training and Evaluating Models ---")
for name, clf in models:
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', clf)
    ])
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    # Evaluate performance
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    results.append({'Model': name, 'Accuracy': acc, 'F1-Score': f1})
    print(f"{name:20} -> Accuracy: {acc:.4f}, F1-Score: {f1:.4f}")

# 2. Display sorted results for comparison
df_results = pd.DataFrame(results).sort_values(by='Accuracy', ascending=False)
display(df_results)
### Hyperparameter Tuning (Grid Search)

The two strongest candidates &ndash; SVM and Random Forest &ndash; are tuned with a 5-fold
cross-validated grid search over their key hyperparameters.
from sklearn.model_selection import GridSearchCV

# We will tune our two best-performing candidates: SVM and Random Forest.

# 1. Define the Parameter Grids
# For SVM: we tune the regularization (C) and the kernel coefficient (gamma)
param_grid_svc = {
    'classifier__C': [0.1, 1, 10, 100],
    'classifier__gamma': ['scale', 'auto', 0.1, 0.01],
    'classifier__kernel': ['rbf', 'poly']
}

# For Random Forest: we tune the number of trees and the depth of the trees
param_grid_rf = {
    #'classifier__n_estimators': [100, 200, 300],
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5, 10]
}

# 2. Setup the Pipelines and GridSearch objects
# Note: SVM requires scaling, while Random Forest is generally invariant to it, 
# but we keep the scaler in the pipeline for consistency.
models_to_tune = [
    ('SVM', SVC(probability=True, random_state=RANDOM_STATE, class_weight='balanced'), param_grid_svc),
    ('Random Forest', RandomForestClassifier(random_state=RANDOM_STATE), param_grid_rf)
]

best_estimators = {}

for name, model, p_grid in models_to_tune:
    print(f"Starting Grid Search for {name}...")
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', model)
    ])
    
    # We use cv=5 (5-fold cross-validation) as suggested in the lectures
    grid_search = GridSearchCV(pipeline, p_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    best_estimators[name] = grid_search.best_estimator_
    
    print(f"--- {name} Results ---")
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best CV Accuracy: {grid_search.best_score_:.4f}\n")

# 3. Final Comparison on the hold-out Test Set
print("--- Final Performance on Test Set ---")
for name, estimator in best_estimators.items():
    y_pred = estimator.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"{name} (Tuned) -> Accuracy: {acc:.4f}, F1-Score: {f1:.4f}")
### Discussion on Grid Search and Hyperparameter Tuning 

The grid search results show that both SVM and Random Forest benefit significantly from the hyperparameter tuning. The SVM with an RBF kernel, with a moderate regularization (C = 10), and adaptive kernel scaling achieved the best overall performance, outperforming the Random Forest on both accuracy and weighted F1-score on the test set.

While both models achieved similar cross-validation accuracy, the tuned SVM generalizes slightly better to unseen data, suggesting that it captures the non-linear structure of the audio features more effectively. Therefore, the SVM was selected as the final model for further evaluation.


### Cross-Validation

The tuned models are re-evaluated with stratified 5-fold cross-validation to assess how
stable their performance is across different data splits.
from sklearn.model_selection import StratifiedKFold, cross_val_score

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

models_cv = {
    "SVM (best)": best_estimators["SVM"],
    "Random Forest (best)": best_estimators["Random Forest"]
}

print("=== Cross-Validation Results (5-Fold) ===")

cv_results = []

for name, model in models_cv.items():
    acc_scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1
    )
    
    f1_scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="f1_weighted",
        n_jobs=-1
    )
    
    print(f"\n{name}")
    print(f"Accuracy: {acc_scores.mean():.3f} ± {acc_scores.std():.3f}")
    print(f"F1-score: {f1_scores.mean():.3f} ± {f1_scores.std():.3f}")
    
    cv_results.append({
        "Model": name,
        "CV Accuracy Mean": acc_scores.mean(),
        "CV Accuracy Std": acc_scores.std(),
        "CV F1 Mean": f1_scores.mean(),
        "CV F1 Std": f1_scores.std()
    })

cv_df = pd.DataFrame(cv_results)
display(cv_df)

### Discussion on Cross-Validation Stability 

The cross-validation results show that both the SVM and Random Forest models achieve a mean accuracy of approximately 74% with relatively low standard deviations across folds. This indicates that the models are stable and their performance is not strongly dependent on a specific train-test split.

Compared to the single hold-out test evaluation, the cross-validation scores are consistent and slightly more reliable.This confirms that the observed performance is representative of the model’s true generalization ability.
The SVM shows marginally lower variance than the Random Forest, suggesting slightly better stability across different subsets of the data.

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

best_model_name = 'SVM' 
final_model = best_estimators[best_model_name]

#predictions on the test set
y_pred = final_model.predict(X_test)


# This shows Precision, Recall, and F1-Score per genre
genre_names = dataset.features['genre'].names
print(f"--- Final Classification Report ({best_model_name}) ---")
print(classification_report(y_test, y_pred, target_names=genre_names))







### Discussion: Metrics and Class Imbalance

As seen in the classification report, the weighted **F1-score** is a more reliable metric
than raw accuracy for this dataset, because it balances precision and recall. If a genre
has high recall but low precision, the model is over-predicting that genre.

The weighted average F1-score is particularly useful here to account for the class
imbalances identified during the initial data exploration.
# 4. Plot the Confusion Matrix
# This helps identify exactly which genres are being confused with each other
plt.figure(figsize=(12, 10))
ConfusionMatrixDisplay.from_estimator(
    final_model, 
    X_test, 
    y_test, 
    display_labels=genre_names, 
    xticks_rotation=45,
    cmap='viridis'
)
plt.title(f'Confusion Matrix: {best_model_name}')
plt.show()
#### Discussion on the confusion matrix: 
The confusion matrix shows that the SVM correctly classifies distinctive genres such as classical and blues, while frequently confusing stylistically similar genres like pop, disco, and rock.
In combination with the learning curve, this pattern seems to be a hint of overfitting, as the model achieves near-perfect training performance but generalizes less effectively to unseen data.
This suggests that the SVM has learned overly complex decision boundaries, and that stronger regularization or feature reduction could improve generalization.

from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(
    pipeline,
    X,
    y,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

plt.figure(figsize=(8,5))
plt.plot(train_sizes, train_scores.mean(axis=1), label="Training")
plt.plot(train_sizes, test_scores.mean(axis=1), label="Validation")
plt.xlabel("Training Samples")
plt.ylabel("Accuracy")
plt.title("Learning Curve SVM")
plt.legend()
plt.grid(True)
plt.show()


### Discussion on the Learning Curve

The learning curve of the SVM shows a strong overfitting behavior, 
with training accuracy close to 1.0 and a large gap to validation accuracy. 
By increasing regularization and reducing feature dimensionality, 
the gap can be reduced, leading to better generalization
