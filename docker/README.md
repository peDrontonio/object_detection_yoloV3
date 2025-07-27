# 🐳 Rodando o Docker Compose: synthetic & vision 🚀


Este README explica passo a passo como levantar os serviços **synthetic** e **vision** usando o seu `docker-compose.yml`.

---

## 1. 🧱Pré‑requisitos

- **Docker** (>= 20.10) e **Docker Compose** (ou o plugin integrado `docker compose`) instalados.  
- **NVIDIA Container Toolkit** para expor GPUs no container (caso use aceleração CUDA).  
- Terminal na pasta onde está o `docker-compose.yml` (supondo: `docker/` na raiz do repo).

---

## 2. 📂 Estrutura do `docker-compose.yml`

```yaml
services:
  synthetic:
    build:
      context: ../synthetic
      dockerfile: ../docker/Dockerfile.blenderproc
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    image: petrobras_synthetic
    container_name: synthetic_container
    volumes:
      - ../shared_data:/app/shared_data
      - ../synthetic:/app
    working_dir: /app
    tty: true
    command: ["bash"]

  vision:
    build:
      context: ../VisionPipelineSuite
      dockerfile: ../docker/Dockerfile.vision
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    image: petrobras_vision
    container_name: vision_container
    volumes:
      - /home/pedrinho:/host-home
      - /home/pedrinho/IC_Petrobras-Repositorio_4/shared_data:/app/shared_data
      - /home/pedrinho/IC_Petrobras-Repositorio_4/VisionPipelineSuite:/app
    working_dir: /app
    tty: true
    command: ["bash"]
```
✅ Dica: Ajuste os caminhos de volume conforme sua organização local.

## 3. 🔨Passo a Passo

### 3.1 Posicione-se na pasta certa

```bash
cd ~/IC_Petrobras-Repositorio_4/docker
```

### 3.2 Build + Up

```bash
docker-compose up --build -d
# ou, se usar o plugin:
docker compose up --build -d
```
- -d: roda em background.

- --build: força rebuild das imagens.

### 3.3 Verificar status
  
```bash 
  docker ps
```
- Você deve ver synthetic_container e vision_container rodando.

## 4. 🔍 Acessando o Container

- Se quiser um shell dentro do container:

```bash
docker exec -it synthetic_container bash
# ou
docker exec -it vision_container bash
```

- Aí já cai em /app com todos os scripts montados

## 5. ⚙️Comandos Úteis

- Parar containers:

```bash
docker exec -it synthetic_container bash
# ou
docker exec -it vision_container bash
```
- Ver logs:

```bash
docker-compose logs -f synthetic
```
- Rebuild só um serviço:

```bash
docker-compose build vision
#ou 
docker-compose build synthetic
```
- Remover volumes não usados (cuidado, apaga dados não versionados):

```bash
docker volume prune
```

## 6. 🐞Troubleshooting
- **GPU** não aparece?
- Confirma se o nvidia-smi funciona no host.
- Reinstala o NVIDIA Container Toolkit:
```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```
- Permissão de pasta?
Se der erro de permissão no volume, ajusta dono/grupo no host:
```bash
sudo chown -R $USER:$USER shared_data synthetic
```
> “Frase motivacional foda.”

