#!/usr/bin/env python3
import blenderproc as bproc
import argparse
import os
import numpy as np
import random
import bpy


def main():
    """Renders synthetic views of a 3D scene using BlenderProc.

    Loads a 3D scene from a .blend, .obj, or .ply file, optionally applies dust to objects,
    configures lighting and camera views, renders images with depth, normals, and segmentation,
    and saves the output in COCO or HDF5 formats (Optional).

    Args:
        scene (str): Path to the scene file (.blend, .obj, or .ply). Defaults to a predefined path.
        output_dir (str): Directory where output data will be saved. Defaults to a predefined path.
        num_views (int): Number of camera views to render. Defaults to 10.

    Raises:
        ValueError: If the scene file extension is unsupported.
    """
    parser = argparse.ArgumentParser(
        description="Render a scene with optional transparent-bg images via BlenderProc."
    )
    parser.add_argument('scene', nargs='?', default="/home/pedrinho/IC_Petrobras-Repositorio_4/synthetic/dataset/objects/ex6.blend",
                        help="Path to .blend, .obj, or .ply scene file.")
    parser.add_argument('output_dir', nargs='?', default="/home/pedrinho/IC_Petrobras-Repositorio_4/synthetic/dataset/output",
                        help="Directory to save outputs.")
    parser.add_argument('--num_views', type=int, default=750,
                        help="Number of camera views to sample.")
    args = parser.parse_args()

    # Initialize BlenderProc
    bproc.init()

    # Load scene based on file extension
    ext = os.path.splitext(args.scene)[1].lower()
    if ext == '.blend':
        objs = bproc.loader.load_blend(args.scene, obj_types=["mesh"])
    elif ext in ['.obj', '.ply']:
        objs = bproc.loader.load_obj(args.scene)
    else:
        raise ValueError(f"Unsupported scene format: {ext}")

    # Assign category IDs and estimate object size (radius from bounding box)
    for idx, obj in enumerate(objs):
        obj.set_cp("category_id", idx + 1)
        bbox = np.array(obj.get_bound_box())
        diameter = np.linalg.norm(bbox.max(axis=0) - bbox.min(axis=0))
        obj.set_cp("radius", diameter / 2.0)

    # Apply dust effect to a subset of objects
    DUST_PROB = 0.33
    for obj in objs:
        if random.random() < DUST_PROB:
            strength = random.uniform(0.3, 0.8)
            scale = random.uniform(0.05, 0.2)
            for mat in obj.get_materials():
                bproc.material.add_dust(mat, strength=strength, texture_scale=scale)

    # Set up lighting
    poi = bproc.object.compute_poi(objs)
    light = bproc.types.Light()
    light.set_type("POINT")
    light.set_location([5, -5, 5])
    light.set_energy(1000)

    # Configure camera resolution
    bproc.camera.set_resolution(512, 512)
    max_radius = max(obj.get_cp("radius") for obj in objs)

    # Clear existing camera frames via Blender API
    cam = bpy.context.scene.camera
    if cam.animation_data:
        cam.animation_data_clear()

    # --- Camera sampling with front-only coverage ---
    # Extract true front direction from object transform
    world_mat = objs[0].get_local2world_mat()
    R = world_mat[:3, :3]
    local_front = np.array([0.0, 0.0, 1.0], dtype=float)
    front_dir = R.dot(local_front)
    front_dir /= np.linalg.norm(front_dir)

    # Sampling parameters
    frontal_ratio = 0.7  # 70% within ±45° cone
    cos_frontal = np.cos(np.deg2rad(45))
    cos_half = np.cos(np.deg2rad(90))  # exclude rear hemispace
    min_dist = max_radius * 1.0
    max_dist = max_radius * 30.0
    partial_offset = max_radius * 0.5

    valid_poses = []  # store transformation matrices
    # Generate until enough views
    while len(valid_poses) < args.num_views:
        dir_cand = np.random.uniform(-1, 1, 3)
        dir_cand /= np.linalg.norm(dir_cand)
        dot_val = np.dot(dir_cand, front_dir)
        # Frontal or general sampling
        if random.random() < frontal_ratio:
            if dot_val < cos_frontal:
                continue
        else:
            if dot_val < cos_half:
                continue
        # Compute position and orientation
        distance = random.uniform(min_dist, max_dist)
        location = poi + dir_cand * distance
        target = poi + np.random.uniform(-partial_offset, partial_offset, size=3)
        rotation = bproc.camera.rotation_from_forward_vec(
            target - location,
            inplane_rot=np.random.uniform(-0.7854, 0.7854)
        )
        cam_mat = bproc.math.build_transformation_mat(location, rotation)
        valid_poses.append(cam_mat)

    # Add sampled poses to camera
    for cam_mat in valid_poses:
        bproc.camera.add_camera_pose(cam_mat)

    # Enable various rendering outputs
    bproc.renderer.enable_depth_output(activate_antialiasing=False)
    bproc.renderer.enable_normals_output()
    bproc.renderer.enable_segmentation_output(map_by=["category_id", "instance", "name"])
    bproc.renderer.set_output_format(enable_transparency=True)

    # Perform rendering
    data = bproc.renderer.render()

    # Write COCO-style annotations
    out_coco = os.path.join(args.output_dir, 'coco_data')
    os.makedirs(out_coco, exist_ok=True)
    bproc.writer.write_coco_annotations(
        out_coco,
        instance_segmaps=data["instance_segmaps"],
        instance_attribute_maps=data["instance_attribute_maps"],
        colors=data["colors"],
        color_file_format="PNG",
        append_to_existing_output=True
    )

if __name__ == "__main__":
    main()
