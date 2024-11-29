import cv2
import numpy as np
#from tensorflow.keras.models import load_model
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from tensorflow import keras



# Cargar el modelo de reconocimiento facial entrenado
#model = load_model('ai_models/facial_recognition_model.h5')
model = keras.models.load_model('ai_models/facial_recognition_model.h5')

def recognize_face(face_image):
    #Detectar y reconocer al usuario en una imagen de la cara
    resized = cv2.resize(face_image, (128, 128))  # Ajusta la imagen
    normalized = resized / 255.0  # Normaliza los valores
    reshaped = np.reshape(normalized, (1, 128, 128, 3))  # Formato de entrada

    # Realiza la predicción
    prediction = model.predict(reshaped)
    user_id = np.argmax(prediction)  # ID del usuario más probable
    confidence = np.max(prediction)  # Confianza de la predicción

    return user_id, confidence