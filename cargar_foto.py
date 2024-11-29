import cv2
import os

def registrar_foto(imagen, id_user):
    # Cargar el clasificador de cara de OpenCV
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Leer la imagen
    img = cv2.imread(imagen)

    # Verificar si la imagen fue cargada correctamente
    if img is None:
        print(f"Error: No se pudo cargar la imagen en {imagen}.")
        return

    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detectar caras en la imagen
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Si se detecta al menos una cara
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            # Recortar la cara de la imagen
            face_img = img[y:y+h, x:x+w]

            # Crear el directorio si no existe dentro de una carpeta con el ID del usuario
            os.makedirs(f'datasets/faces/{id_user}', exist_ok=True)

            # Guardar la imagen de la cara recortada
            face_file_name = f'datasets/faces/{id_user}/user_{id_user}.jpg'  # Cambia el nombre si necesitas
            cv2.imwrite(face_file_name, face_img)
            print(f"Foto registrada exitosamente: {face_file_name}")
            break  # Solo guardamos la primera cara detectada
    else:
        print("No se detectó ninguna cara en la imagen.")

# Registrar una foto de un usuario
foto = "otros/IMG_20220523_073027.jpg"
registrar_foto(foto, 5)