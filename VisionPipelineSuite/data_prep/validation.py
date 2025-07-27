import os
from ultralytics import YOLO

# Caminho pro modelo treinado (.pt)
modelo_treinado = "/home/pedrinho/VisionPipelineSuite/data_prep/runs/detect/train/weights/best.pt"  # ajusta o caminho se tiver diferente

# Pasta com imagens que você quer testar
pasta_imagens = "/home/pedrinho/VisionPipelineSuite/dataset/mn/Manometros/Output/images/test"

# Pasta onde os resultados serão salvos
pasta_saida = "/home/pedrinho/VisionPipelineSuite/dataset/mn/Manometros/validacao"

# Cria a pasta de saída se não existir
os.makedirs(pasta_saida, exist_ok=True)

# Carrega o modelo treinado
model = YOLO(modelo_treinado)

# Roda a predição em todas as imagens da pasta e salva os resultados
results = model.predict(
    source=pasta_imagens,
    save=True,
    project=pasta_saida,
    name="",  # evita criar subpastas como 'predict' ou 'exp'
    exist_ok=True  # sobrescreve se já existir
)

print(f"Resultados salvos em: {pasta_saida}")
