import cv2
from ultralytics import YOLO  # Supondo que você esteja usando o Ultralytics YOLO

# Carrega o modelo treinado
model = YOLO("caminho/para/o/modelo/best.pt")  # Substitua pelo caminho do seu checkpoint

# Inicia a captura da câmera
cap = cv2.VideoCapture(0)  # 0 para a câmera padrão

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Executa a inferência
    results = model(frame)
    
    # Desenha as caixas (a API do modelo geralmente tem funções para isso)
    annotated_frame = results[0].plot()  # Exemplo de como obter o frame anotado
    
    # Exibe o frame
    cv2.imshow("Inferência em tempo real", annotated_frame)
    
    # Encerra se a tecla 'q' for pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
