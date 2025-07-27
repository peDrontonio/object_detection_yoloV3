import json
import os
import shutil
import uuid  # Biblioteca opcional caso queira gerar identificadores únicos

def consolidate_coco(json_paths, images_dirs, output_json_path, output_images_dir):
    # Estrutura final para o dataset consolidado conforme padrão COCO
    consolidated = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    # Contadores globais para imagens e anotações
    global_img_id = 1
    global_ann_id = 1

    # Dicionário para mapear nomes de categorias para um novo ID único.
    cat_name_to_id = {}
    
    # Itera sobre cada JSON e seu diretório de imagens correspondente
    for idx, json_path in enumerate(json_paths):
        # Carrega o JSON atual
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Mapeamento local para as categorias deste JSON (ID original -> novo ID)
        local_cat_mapping = {}
        for cat in data["categories"]:
            cat_name = cat["name"]
            # Se a categoria já foi adicionada, usa o mesmo novo ID
            if cat_name in cat_name_to_id:
                local_cat_mapping[cat["id"]] = cat_name_to_id[cat_name]
            else:
                new_cat_id = len(cat_name_to_id) + 1  # Gera um novo ID para a categoria
                cat_name_to_id[cat_name] = new_cat_id
                local_cat_mapping[cat["id"]] = new_cat_id
                # Adiciona a categoria ao dataset consolidado com o novo ID
                new_cat = cat.copy()
                new_cat["id"] = new_cat_id
                consolidated["categories"].append(new_cat)
        
        # Mapeamento local para as imagens deste JSON (ID original -> novo ID)
        local_img_mapping = {}
        
        # Processamento das imagens
        for img in data["images"]:
            old_img_id = img["id"]
            new_img_id = global_img_id
            global_img_id += 1

            # Gera um novo nome de arquivo para evitar conflitos
            original_filename = img["file_name"]
            new_filename = f"{new_img_id}_{original_filename}"
            
            # Atualiza os dados da imagem
            new_img = img.copy()
            new_img["id"] = new_img_id
            new_img["file_name"] = new_filename
            consolidated["images"].append(new_img)
            local_img_mapping[old_img_id] = new_img_id
            
            # Caminhos para copiar a imagem
            source_img_path = os.path.join(images_dirs[idx], original_filename)
            destination_img_path = os.path.join(output_images_dir, new_filename)
            if os.path.exists(source_img_path):
                shutil.copy2(source_img_path, destination_img_path)
            else:
                print(f"Aviso: imagem {source_img_path} não encontrada.")
        
        # Processamento das anotações
        for ann in data["annotations"]:
            new_ann = ann.copy()
            new_ann["id"] = global_ann_id
            global_ann_id += 1

            # Atualiza o image_id com base no mapeamento local
            old_img_id = ann["image_id"]
            if old_img_id in local_img_mapping:
                new_ann["image_id"] = local_img_mapping[old_img_id]
            else:
                print(f"Aviso: Anotação com image_id {old_img_id} não encontrada no mapeamento.")
            
            # Atualiza o category_id com base no mapeamento local
            old_cat_id = ann["category_id"]
            if old_cat_id in local_cat_mapping:
                new_ann["category_id"] = local_cat_mapping[old_cat_id]
            else:
                print(f"Aviso: Anotação com category_id {old_cat_id} não encontrada no mapeamento.")
            
            consolidated["annotations"].append(new_ann)
    
    # Garante que o diretório de saída para as imagens exista
    os.makedirs(output_images_dir, exist_ok=True)

    # Salva o arquivo JSON consolidado
    with open(output_json_path, "w", encoding="utf-8") as out_file:
        json.dump(consolidated, out_file, ensure_ascii=False, indent=4)
    
    print("Consolidação concluída com sucesso.")

if __name__ == "__main__":
    # Exemplos de entradas – substitua com os caminhos reais do seu ambiente
    json_paths = [
        "/home/pedrinho/VisionPipelineSuite/dataset/mn/mn01/output/instances_coco_output.json",
        "/home/pedrinho/VisionPipelineSuite/dataset/mn/mn03/output/instances_coco_output.json",
        "/home/pedrinho/VisionPipelineSuite/dataset/mn/mn04/output/instances_coco_output.json"
    ]
    
    images_dirs = [
        "/home/pedrinho/VisionPipelineSuite/dataset/mn/mn01/images/default",
        "/home/pedrinho/VisionPipelineSuite/dataset/mn/mn03/images/default",
        "/home/pedrinho/VisionPipelineSuite/dataset/mn/mn04/images/default"
    ]
    
    output_json_path = "/home/pedrinho/VisionPipelineSuite/dataset/mn/Manometros/final.jsons"
    output_images_dir = "/home/pedrinho/VisionPipelineSuite/dataset/mn/Manometros/images"

    consolidate_coco(json_paths, images_dirs, output_json_path, output_images_dir)
