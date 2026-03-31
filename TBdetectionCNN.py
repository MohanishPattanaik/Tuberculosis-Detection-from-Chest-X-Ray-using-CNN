# Import libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import cv2
import io
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import ipywidgets as widgets
from IPython.display import display, clear_output

import kagglehub

path = kagglehub.dataset_download("tawsifurrahman/tuberculosis-tb-chest-xray-dataset")

print("Path to dataset files:", path)

# Dataset path (from Kaggle)
TBpath  = "/root/.cache/kagglehub/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset/versions/3/TB_Chest_Radiography_Database/Tuberculosis"
Normalpath  = "/root/.cache/kagglehub/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset/versions/3/TB_Chest_Radiography_Database/Normal"
base_path = "/root/.cache/kagglehub/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset/versions/3/TB_Chest_Radiography_Database"

print("Number of TB images:", len(os.listdir(TBpath)))
print("Number of Normal images:", len(os.listdir(Normalpath)))

def show_examples(class_path, title, n=5):
    fig, axes = plt.subplots(1, n, figsize=(15,5))
    for i, fname in enumerate(os.listdir(class_path)[:n]):
        img = cv2.imread(os.path.join(class_path, fname))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(title)
    plt.show()

show_examples(Normalpath, "Normal")
show_examples(TBpath, "Tuberculosis")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.15,
    horizontal_flip=True,
    zoom_range=0.1
)

train_data = train_datagen.flow_from_directory(
    base_path,
    target_size=(224,224),
    class_mode='binary',
    batch_size=32,
    subset='training'
)

val_data = train_datagen.flow_from_directory(
    base_path,
    target_size=(224,224),
    class_mode='binary',
    batch_size=32,
    subset='validation'
)

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(train_data, validation_data=val_data, epochs=10)

model.save("tuberculosis_model.h5")

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.legend()
plt.show()
