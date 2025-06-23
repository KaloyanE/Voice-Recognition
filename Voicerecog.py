import os	
import numpy as np
import librosa	from sklearn.model_selection
import train_test_split
from sklearn.svm import SVC	from sklearn.metrics 
import accuracy_score, classification_report
# Извеждане на MFCC характеристики
def extract_mfcc(file_path, n_mfcc=40, max_len=174):
    audio, sr = librosa.load(file_path, sr=16000)  # Зарежда аудио с честота 16kHz
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)  
    # Допълва се с нули или се изрязва част от файла, за еднаква дължина на данните
    if mfccs.shape[1] < max_len:
        pad_width = max_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :max_len]
    return mfccs.flatten()  # преобразува се до едноизмерен масив
# Зареждане на набор от данни от директория
def load_dataset(dataset_path):
    X, y = [], []
    label_map = {}
    for label, person_folder in enumerate(sorted(os.listdir(dataset_path))):
        label_map[label] = person_folder
        person_path = os.path.join(dataset_path, person_folder)    
        # Преминава през всички файлове на поддиректориите
        for file in os.listdir(person_path):
            if file.endswith('.mp3'):  # Работа с mp3 файлове
                file_path = os.path.join(person_path, file)
                features = extract_mfcc(file_path)
                X.append(features)
                y.append(label)       
    return np.array(X), np.array(y), label_map
# Трениране на SVM 
def train_model(X_train, y_train):
    model = SVC(kernel='linear', probability=True)  # Support Vector Classifier с линейно ядро
    model.fit(X_train, y_train)
    return model
# Оценка на модела
def evaluate_model(model, X_test, y_test, label_map):
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred, target_names=label_map.values()))
# Предположение чии е новия аудио файл
def predict_person(file_path, model, label_map):
    features = extract_mfcc(file_path).reshape(1, -1)
    prediction = model.predict(features)
    return label_map[prediction[0]]
dataset_path = "X:\\projects\\KP II\\Dataset"
# Зареждане на набор от данни
X, y, label_map = load_dataset(dataset_path)
print(f"Loaded {X.shape[0]} samples across {len(label_map)} classes.")
# Разделяне на данните
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42, stratify=y)
# Трениране на SVM модел
model = train_model(X_train, y_train)
# Оценка на модела
evaluate_model(model, X_test, y_test, label_map)
# Тестване на нов аудио файл
new_sample = "X:\\projects\\KP II\\test\\Person 10 test.mp3"  
predicted_person = predict_person(new_sample, model, label_map)
print(f"The voice belongs to: {predicted_person}")
