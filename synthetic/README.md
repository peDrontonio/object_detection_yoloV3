# 📚 Dataset Sintético

Este README explica a organização dos dados gerados pelo **BlenderProc** e como visualizar as anotações COCO e arquivos HDF5. Para rodar os scripts Python e o Docker, veja os READMEs nas pastas correspondentes:

- 🔧 [Instruções Python & BlenderProc](https://github.com/EESC-LabRoM/IC_Petrobras-Repositorio_4/tree/main/synthetic#readme)  
- 🐳 [Instruções Docker Compose](https://github.com/EESC-LabRoM/IC_Petrobras-Repositorio_4/blob/main/docker/README.md)  

---

## 📁 Organização de Pastas

Após executar o pipeline, seu `output_dir` terá a estrutura:

- Cada `run_<i>` corresponde a uma configuração de cena diferente.  
- Dentro de `coco_data/`, você encontra o JSON e as imagens segmentadas para uso direto em frameworks como Detectron2 ou MMDetection.  
- O `data.hdf5` inclui **todos** os mapas gerados:
  - `colors`  (RGB)
  - `depth`   (profundidade)
  - `normals` (normais de superfície)
  - `instance_segmaps` e `instance_attribute_maps`

---
## Visualizacao dos arquivos HDF5 e COCO

- Visualize os arquivos HDF5:

```bash
blenderproc vis hdf5 caminho/paraoutput/0.hdf5
```

- Para os aquivos COCO, execute:

```bash
blenderproc vis coco [-i <image index>] [-c <coco annotations json>] [-b <base folder of coco json and image files>]
```

- Dentro do diretorio:

```bash
blenderproc vis coco -i 1 -c coco_annotations.json -b caminhos/para/output/coco_data
```

## 📝 Anotações COCO

O `instance_annotations.json` segue a especificação COCO:

- **images**: lista de metadados (altura, largura, caminho)  
- **annotations**: bounding boxes, masks (via `instance_segmaps`), category_id e instance_id  
- **categories**: IDs e nomes de cada objeto  

🔍 **Dica**: Você pode carregar e inspecionar com Python:

```python
import json
from pathlib import Path

coco = json.load(open('run_0/coco_data/annotations.json'))
print("Categorias:", coco['categories'])
print("Total de imagens:", len(coco['images']))
print("Total de anotações:", len(coco['annotations']))
```

