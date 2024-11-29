import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Ruta del dataset (cada carpeta representa un usuario)
DATASET_PATH = 'datasets/faces/'  # Estructura: datasets/faces/user_id/

def load_data():
    images = []
    labels = []
    for user_folder in os.listdir(DATASET_PATH):
        user_id = int(user_folder)  # Cada carpeta es el ID del usuario
        folder_path = os.path.join(DATASET_PATH, user_folder)
        #imprimir la ruta de la carpeta
        print(folder_path)
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            img = load_img(img_path, target_size=(128, 128))
            img_array = img_to_array(img) / 255.0  # Normaliza los valores
            images.append(img_array)
            labels.append(user_id)
            #imprimir la ruta de la imagen
            print(img_path)

    return np.array(images), to_categorical(np.array(labels))

# Cargar datos
X, y = load_data()

# Definir el modelo de reconocimiento facial
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(y.shape[1], activation='softmax')  # Salida con número de usuarios
])

# Compilar y entrenar el modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=10, batch_size=32)

# Guardar el modelo entrenado
model.save('ai_models/facial_recognition_model.h5')
print("Modelo guardado exitosamente.")