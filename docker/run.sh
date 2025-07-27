# #!/bin/bash

# set -e

# # Diretório do script
# cd "$(dirname "$0")"

# # Verifica o parâmetro passado
# case "$1" in
#   up)
#     echo "🚀 Subindo containers no modo interativo..."
#     docker-compose up
#     ;;
#   up-bg)
#     echo "🚀 Subindo containers em background..."
#     docker compose up -d
#     ;;
#   down)
#     echo "🧹 Derrubando containers..."
#     docker compose down
#     ;;
#   restart)
#     echo "🔁 Reiniciando containers..."
#     docker compose down
#     docker compose up -d
#     ;;
#   logs)
#     docker compose logs -f
#     ;;
#   *)
#     echo "❓ Uso: $0 {up | up-bg | down | restart | logs}"
#     ;;
# esac

#!/bin/bash
# Uso: ./run.sh [nome_servico]
# if [ -z "$1" ]; then
#   docker compose up --build
# else
#   docker compose up --build "$1"
# fi
#!/bin/bash

# Inicia o container "vision" com suporte à GPU
#!/bin/bash

# Remove o container "vision_container" se existir
if [ "$(docker ps -aq -f name=vision_container)" ]; then
  echo "Removendo container existente: vision_container"
  docker rm -f vision_container
fi

# Converte os caminhos relativos para absolutos
SHARED_DATA=$(realpath ../shared_data)
VISION_PIPELINE=$(realpath ../VisionPipelineSuite)
SYNTHETIC=$(realpath ../synthetic)

# Inicia o container petrobras_vision com suporte à GPU usando os caminhos absolutos
docker run --rm --gpus all -it \
  --name vision_container \
  -v "${SHARED_DATA}:/app/shared_data" \
  -v "${VISION_PIPELINE}:/app" \
  -w /app \
  petrobras_vision \
  bash
