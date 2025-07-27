import blenderproc as bproc
import argparse
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('pix_path', nargs='?', default="/home/pedrinho/IC_Petrobras-Repositorio_4/projeto_petrobras/dataset/pix3d",
                    help="Caminho para o dataset Pix3D")
parser.add_argument('cc_material_path', nargs='?', default="/home/pedrinho/IC_Petrobras-Repositorio_4/projeto_petrobras/dataset/cc_textures",
                    help="Caminho para a pasta CCTextures")
parser.add_argument('output_dir', nargs='?', default="/home/pedrinho/IC_Petrobras-Repositorio_4/projeto_petrobras/dataset/output",
                    help="Diretório de saída")
args = parser.parse_args()

bproc.init()

# Carrega materiais
materials = bproc.loader.load_ccmaterials(
    args.cc_material_path, ["Bricks", "Wood", "Carpet", "Tile", "Marble"]
)

# Carrega objetos do Pix3D (exemplo: categoria "bed")
interior_objects = []
for i in range(15):
    objs = bproc.loader.load_pix3d(data_path=args.pix_path, used_category="table")
    interior_objects.extend(objs)

# Atribui IDs de categoria (exemplo simples)
for j, obj in enumerate(interior_objects):
    obj.set_cp("category_id", j + 1)

# Constrói sala aleatória
objects = bproc.constructor.construct_random_room(
    used_floor_area=25,
    interior_objects=interior_objects,
    materials=materials,
    amount_of_extrusions=5
)

# Iluminação simples
bproc.lighting.light_surface(
    [obj for obj in objects if obj.get_name() == "Ceiling"],
    emission_strength=4.0, emission_color=[1,1,1,1]
)

# Cria BVH
bvh_tree = bproc.object.create_bvh_tree_multi_objects(objects)
floor = [obj for obj in objects if obj.get_name() == "Floor"][0]

# Amostragem de poses
poses = 0
tries = 0
while tries < 10000 and poses < 5:
    location = bproc.sampler.upper_region(floor, min_height=1.5, max_height=1.8)
    rotation = np.random.uniform([1.0, 0, 0], [1.4217, 0, 6.283185307])
    cam2world_matrix = bproc.math.build_transformation_mat(location, rotation)
    if (bproc.camera.perform_obstacle_in_view_check(cam2world_matrix, {"min": 1.2}, bvh_tree)
        and bproc.camera.scene_coverage_score(cam2world_matrix) > 0.4
        and floor.position_is_above_object(location)):
        bproc.camera.add_camera_pose(cam2world_matrix)
        poses += 1
    tries += 1

for materials in obj.get_materials():
    bproc.material.add_dust(materials, strength=0.8, texture_scale=0.05)

# Ativa saídas
bproc.renderer.enable_depth_output(activate_antialiasing=False)
bproc.renderer.enable_normals_output()
bproc.renderer.enable_segmentation_output(map_by=["category_id", "instance", "name"], default_values={'category_id':0})
bproc.renderer.set_light_bounces(max_bounces=200, diffuse_bounces=200, glossy_bounces=200,
                                 transmission_bounces=200, transparent_max_bounces=200)

# Render
data = bproc.renderer.render()

# Salva anotações COCO
bproc.writer.write_coco_annotations(
    os.path.join(args.output_dir, 'coco_data'),
    instance_segmaps=data["instance_segmaps"],
    instance_attribute_maps=data["instance_attribute_maps"],
    colors=data["colors"],
    color_file_format="JPEG"
)

# Salva em HDF5
bproc.writer.write_hdf5(args.output_dir, data)
