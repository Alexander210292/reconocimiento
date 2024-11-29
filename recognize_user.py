import cv2
#from face_recognition import recognize_face
import requests
import base64
import json
import config as cfg

# crear una función para consumiar la api de reconocimiento facial http://127.0.0.1:8000/api/recognize_face/ 
# con la imagen de la cara "faces" y devolver el id del usuario y la confianza
# si la confianza es mayor a 0.7, se considera que el usuario ha sido reconocido
# si la confianza es menor a 0.7, se considera que el usuario no ha sido reconocido

def api_recognize_face(face):
    ENDPOINT = cfg.SERVER + "api/recognize_face/"
    try:
         # Convertir la imagen a un buffer
        _, buffer = cv2.imencode('.jpg', face)
        files = {
            "face_image": ("face.jpg", buffer.tobytes(), "image/jpeg")  # Clave y archivo en memoria
        }

        # Solicitud POST
        response = requests.post(
            ENDPOINT,
            files=files
        )
        
        # Mostrar el payload enviado para depuración
        #print("Payload enviado:", json.dumps(payload, indent=2)[:500]) 

        # Enviar la solicitud POST
       
        response.raise_for_status()  # Lanza una excepción para errores HTTP
        
        # Validar y retornar la respuesta
        data = response.json()
        
        if data.get('confidence') > 0.7:
            return data

    except requests.exceptions.RequestException as e:
        print("Error al registrar la transacción:", e)
        return None
    except (json.JSONDecodeError, KeyError) as e:
        print("Error al procesar la respuesta del servidor:", e)
        return None
    

def recognize_user():
    success = 0
    user_id = None
    confidence = 0.0
    # Inicia la cámara
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detecta caras usando Haar Cascade
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = frame[y:y + h, x:x + w]
 
            # Llama a la API usando el archivo guardado
            result = api_recognize_face(face)
            
            if result:
                user_id = result.get('user_id')
                confidence = result.get('confidence')
            
            #print(json_date)
            
            #if confidence > 0.8:
            #print(f"Usuario: {user_id}, Confianza: {confidence:.2f}")
            #else:
            #    print("Desconocido")

            # Dibujar un rectángulo en la cara detectada
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if confidence > 0.7:
                cv2.putText(frame, f"ID: {user_id} - {success}%", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                success = success + 5 
            else:
                cv2.putText(frame, "Desconocido", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            
            if success > 100:
                cap.release()
                cv2.destroyAllWindows()
                return user_id, confidence, face

        cv2.imshow("Reconocimiento Facial", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
    return user_id, confidence, face