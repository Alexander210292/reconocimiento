from recognize_user import recognize_user
import requests
import base64
import cv2
import torch
from torchvision import models, transforms
from PIL import Image
import config as cfg
# Arduino
#import serial
#import time

#Configuracion de arduino
#ser = serial.Serial('COM8', 9600, timeout=1)
#time.sleep(2)

# Configuración de los endpoints
LOGIN_ENDPOINT = cfg.LOGIN_ENDPOINT
TRANSACTION_ENDPOINT = cfg.TRANSACTION_ENDPOINT

# Configuración del modelo PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Reconstruir el modelo y cargar los pesos
model = models.resnet18(pretrained=False)  # Cambia según la arquitectura usada
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, 1)  # Ajusta según tu modelo
model.load_state_dict(torch.load("ai_models/residuo_model.pth", map_location=device))
model = model.to(device)
model.eval()

# Transformaciones para el modelo
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def trasaction_new(token, user_id, waste_type_id, container_id, points_awarded):
    """Registra una nueva transacción en el sistema."""
    try:
        payload = {
            "user_id": int(user_id),
            "waste_type_id": int(waste_type_id),
            "container_id": container_id,
            "points_awarded": points_awarded,
        }
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        response = requests.post(TRANSACTION_ENDPOINT, json=payload, headers=headers)
        response.raise_for_status()  # Lanza una excepción para errores HTTP
        print("Transacción registrada correctamente:", response.json())
    except requests.exceptions.RequestException as e:
        print("Error al registrar la transacción:", e)

def encode_image_to_base64(image):
    """Codifica una imagen en formato base64."""
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

def send_image_to_endpoint():
    """Envía la imagen codificada al endpoint de login."""
    headers = {'Content-Type': 'application/json'}
    data = {'username': 'wfarel', 'password': 'wf12345*'}
    
    try:
        response = requests.post(LOGIN_ENDPOINT, json=data)
        if response.status_code == 200:
            print("Usuario reconocido:", response.json())
            return response.json().get('access')
        else:
            print(f"Error: {response.status_code}, {response.text}")
    except requests.exceptions.RequestException as e:
        print("Error al enviar la imagen:", e)
    return None

def predict_waste_type(frame):
    """Clasifica el tipo de residuo usando el modelo PyTorch."""
    try:
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        input_tensor = data_transforms(pil_image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor).squeeze()
            pred = torch.sigmoid(output).item()
            prediction = "plastic" if pred > 0.5 else "metal"
            points_awarded = 1000 if prediction == "metal" else 500
            waste_type_id = 2 if prediction == "metal" else 1
            return prediction, points_awarded, waste_type_id
    except Exception as e:
        print("Error al procesar el residuo:", e)
        return None, 0, 0

def capture_and_classify_waste(token, user_id):
    """Lanza la cámara para clasificar residuos y registrar transacciones."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No se pudo acceder a la cámara.")
        return

    print("Presiona 's' para capturar y clasificar un residuo. Presiona 'q' para salir.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error al capturar el frame.")
            break
        
        # Mostrar el feed de la cámara
        cv2.imshow("Identificador de Residuos", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Salir
            break
        elif key == ord('s'):  # Clasificar residuo
            prediction, points_awarded, waste_type_id = predict_waste_type(frame)
            if waste_type_id:
                print(f"Residuo clasificado: {prediction}, Puntos: {points_awarded}")
                 #if prediction == "metal":
                    #ser.write(b'n')
                 #else:
                    #ser.write(b'p')
                trasaction_new(token, user_id, waste_type_id, 1, points_awarded)

    cap.release()
    cv2.destroyAllWindows()
    #ser.close()

def main():
    # Autenticación del usuario
    user_id, confidence, image = recognize_user()
    print(f"Usuario: {user_id}, Confianza: {confidence:.2f}")
    
    #encoded_image = encode_image_to_base64(image)
    token = send_image_to_endpoint()
    print(f"Token: {token}")
    
    if not token or not user_id:
        print("Error: Token o user_id no válidos.")
        return
    
    #Lanzar la cámara para clasificar residuos
    capture_and_classify_waste(token, user_id)

    print("Fin del programa.")

if __name__ == "__main__":
    main()
