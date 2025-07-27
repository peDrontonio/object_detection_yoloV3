# 🧠 Identificação e Leitura de Sensores de Manometria com IA

## 📌 Visão Geral

Este projeto tem como objetivo o desenvolvimento de um sistema automatizado para **identificação e interpretação de sensores de manometria** em instalações de óleo e gás, utilizando **dados sintéticos** para treinar modelos de machine learning capazes de realizar **monitoramento de pressão em tempo real** com robôs autônomos.

Apesar do avanço da robótica industrial, a escassez de datasets diversificados e de alta qualidade para leitura de sensores ainda é um gargalo. Este projeto resolve isso através da **geração procedural de dados sintéticos**, simulando diferentes condições ambientais (iluminação, oclusão, variação operacional etc).

As imagens geradas são anotadas com precisão para leitura numérica dos mostradores, permitindo o treinamento de modelos robustos com o algoritmo **YOLOv11**. A integração é feita com plataformas como a **ANYmal C**, que navega com facilidade em ambientes complexos.

### ✅ Benefícios
- Redução da exposição humana a ambientes perigosos
- Detecção precoce de falhas e vazamentos
- Otimização da manutenção via predição
- Adaptação a outros setores industriais (químico, energia etc)

---

## 🗂 Estrutura do Projeto

Abaixo está a estrutura geral de diretórios do repositório:

![Estrutura de pastas](./image.png)

```bash
IC_Petrobras-Repositorio_4/
├── docker/
│   ├── docker-compose.yml
│   ├── vision/               # Treinamento com YOLOv11
│   └── blenderproc/          # Geração de dados sintéticos
├── projeto_petrobras/        # Documentação e scripts para geração de dados sintéticos
├── shared_data/              # Pasta compartilhada entre containers
├── VisionPipelineSuite/      # Pipeline de visão computacional
└── README.md
```

## 🐳 Como Rodar com Docker Compose

### ✅ Pré-requisitos

- Docker e Docker Compose (v2 ou superior)
- Suporte à GPU (instale `nvidia-container-toolkit` e configure o runtime)

### 🔁 Rodar todos os containers de uma vez:

```bash
cd docker
docker compose up --build

```
-Se aparecer o erro:

```lua
could not select device driver "" with capabilities: [[gpu]]
```

-⚙️ Execute:

```bash 
sudo nvidia-ctk runtime configure --runtime=nvidia
sudo systemctl restart docker
```
### ▶️ Rodar containers individualmente:
  
  - 🔧 Geração de dados (BlenderProc)
  ```bash 
  docker compose run --rm synthetic_container
  ```
  - 🧠 Pipeline de visão (YOLOv11)
  ```bash
  docker compose run --rm vision_container
  ```
## 🤖 Como Rodar os Scripts de Criação dos Dados Sintéticos

### ✅ Pré‑requisitos

- Ter o contêiner Docker em execução  
- Disponibilizar os arquivos `.blend` ou `.obj` que serão usados

### 🛠️ Uso do Script Python

```bash
python main.py \
  path/to/scene.blend \
  path/to/hdri_folder \
  path/to/output_dir \
  --runs 3 \
  --views_per_run 5
```
- 📚Para explicacoes mais detalhadas
- [Synthetic Data](https://github.com/pedrinho/IC_Petrobras-Repositorio_4/synthetic/README.md)

## Contribuições:

Encontrou um erro? nos ajude a consertar. Se quiser implementar uma nova funcionalidade ou uma melhoria dentro do projeto faça um pedido de PR.

Faça o fork e clone o repositório:

```bash
git clone https://github.com/seu-usuario/IC_Petrobras-Repositorio_4.git
cd IC_Petrobras-Repositorio_4
```

Crie uma nova branch:

```bash
git checkout -b feature/nome-da-feature
```

Teste suas modificações

Faça commit e push:

```bash
git add .
git commit -m "feat: nova feature"
git push origin feature/nome-da-feature
```
Crie um Pull Request no GitHub ✅


## 💬 Contato

Desenvolvido com 💡 por [@peDrontonio](https://github.com/peDrontonio) e [@EnzoFMK](https://github.com/EnzoFMK)

Organização: [@LabRoM-USP](https://github.com/LabRoM-USP)

Entre em contato via GitHub para dúvidas, sugestões ou parcerias.

> “Segurança não é luxo, é necessidade. E com IA, ela é inteligente.”
