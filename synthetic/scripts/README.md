# 🚀 Geração de Dados Sintéticos com BlenderProc 🎨

## 1. Visão Geral 🔍  
Este projeto automatiza a criação de datasets sintéticos usando o BlenderProc, gerando múltiplas cenas com variação de materiais, aplicação de poeira procedural e fundos HDRI aleatórios. Ele produz:

- **Variação PBR** nos objetos (Specular IOR, Roughness, Metallic, Base Color) 🖌️  
- **Poeira procedural** com probabilidade customizável 🌫️  
- **Iluminação dinâmica** via luz pontual e mapas HDRI de ambiente 💡  
- **Saídas** em formato COCO (JSON + JPEG) 📦 e HDF5 💾  

## 2. Pré‑requisitos 📋  
- **Blender 3.x** e **BlenderProc** instalado 🐳  
- **Python 3.8+** com as bibliotecas:
  - `numpy` ➕
  - `argparse` 🛠️
  - `blenderproc` 🔧
- Arquivos de entrada:
  - Um `.blend` (ou `.obj` + cena no Blender) contendo seus modelos 📂  
  - Uma pasta de imagens HDRI para background 🌆  
- **Docker Compose** (opcional, para rodar em container) 🐳

## 3. Instalação ⚙️  

### 3.1. Local (sem Docker) 💻  
1. Crie um ambiente virtual ou Conda:
   ```bash
   python -m venv venv
   source venv/bin/activate
2. Instale dependências:
```conda create env -f eniroment.yml
```
### 3.2. Com Docker Compose 🐳
1. Na raiz do repo, entre na pasta docker/:
```bash
cd docker
```
2. Suba os containers:
```bash
docker-compose up --build -d
```
3. Verifique se os serviços synthetic e vision estão rodando:
```bash
docker ps
```
4. Entre no container synthetic para rodar o script:
```bash
docker exec -it synthetic_container bash
```
📖 Para mais detalhes sobre o Docker Compose:
[Docker Compose](/home/pedrinho/IC_Petrobras-Repositorio_4/docker/README.md)

## 4. Uso Basico 🚀
- Dentro do ambiente (ou container) execute:

```bash
python main.py \
  /caminho/para/scene.blend \
  /caminho/para/hdri_folder \
  /caminho/para/output_dir \
  --runs 3 \
  --views_per_run 5
```
- scene: arquivo .blend que contém seus objetos 🖼️

- hdri_path: pasta com HDRIs 🌄

- output_dir: onde salvar run_<i>/ 📁

- --runs: número de cenários (default 3) 🔄

- --views_per_run: câmeras por cenário (default 3) 📸

### 4.1. Parâmetros Úteis ⚗️
Você pode ajustar dentro do código (ou expor via flags adicionais):
- dust_prob (probabilidade de poeira, ex.: 0.33) 🐚
- dust_strength_range e dust_scale_range (intervalos para variação de poeira) 🎚️
- Resolução da câmera em bproc.camera.set_resolution(w, h) 🖥️

## 5. Estrutura de Saída 🗂️
Após rodar, o output_dir ficará com algo como:
```kotlin
output_dir/
├─ run_0/
│  ├─ coco_data/
│  │  ├─ annotations.json
│  │  ├─ rgb_0000.jpg
│  │  └─ ...
│  └─ data.hdf5
├─ run_1/
│  └─ ...
└─ run_2/
   └─ ...
```
- Cada run_<i> contém os assets no formato COCO e o HDF5 completo 🎯

- Organize ou combine os JSONs conforme sua pipeline de treinamento 🏋️

## 6. Exemplos e Imagens🖼️

- ![Colocar uma fotos de dados gerados aqui]()
*Figura 1: Inserir legenda.*

✨ Dicas finais

- Para rodar headless (sem GUI), use a flag --background no BlenderProc 🤖

- Ajuste a distância da câmera e o ponto de interesse (compute_poi) para focar no objeto principal 🎯

- Experimente diferentes HDRIs para maximizar diversidade de iluminação 🌈