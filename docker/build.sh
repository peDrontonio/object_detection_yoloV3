# #!/bin/bash

# set -e  # Encerra se ocorrer algum erro

# echo "🔧 Iniciando build do docker-compose..."

# # Vai para o diretório onde está este script
# cd "$(dirname "$0")"

# # Build dos serviços definidos no docker-compose.yml
# docker compose build

# echo "✅ Build concluído com sucesso!"

#!/bin/bash
# Uso: ./build.sh [nome_servico]
if [ -z "$1" ]; then
  docker compose build
else
  docker compose build "$1"
fi
