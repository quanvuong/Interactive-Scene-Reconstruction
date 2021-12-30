import numpy as np
import pandas as pd
import open3d as o3d
from utils import *

# Rely on sapien for loading object visual mesh since PartNet mesh is separated for object parts
import sys
sys.path.append('/home/kolinguo/Desktop/Research/code')
import sapien.core as sapien
import gym
import mobile_pnp.env
env = gym.make('ArmGraspObjectClutter_0-v0')

from sapien.core import Pose
from sapien.asset import download_partnet_mobility
from mobile_pnp.utils.download_assets import download_shapenet

import yaml
from pathlib import Path

output_dataset_folder = (Path(__file__) / '../../../cad_dataset_sapien/object_mesh').resolve()

def load_articulation(env, articulation_config: dict) -> sapien.Articulation:
    """Load a PartNet or ShapeNet object"""
    TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6InJ1Z3VvQHVjc2QuZWR1IiwiaXAiOiIxNzIuMjAuMC4xIiwicHJpdmlsZWdlIjoxLCJpYXQiOjE2Mjc3MTEzODgsImV4cCI6MTY1OTI0NzM4OH0.SAUjbW2x7XJXRAOwq8FZB1116IHLLX5Av_PrBvTy8NE"

    if 'partnet_mobility_id' in articulation_config:
        urdf = download_partnet_mobility(
            articulation_config['partnet_mobility_id'],
            token=TOKEN,
            directory=None
        )
        vhacd_urdf = Path(urdf).parent.joinpath('mobility_fixed.urdf')
        if vhacd_urdf.exists():
            urdf = str(vhacd_urdf)
    elif 'shapenet_synset_model_id' in articulation_config:
        synsetId_modelId = articulation_config['shapenet_synset_model_id']
        urdf = download_shapenet(synsetId_modelId, directory=None)
        vhacd_urdf = Path(urdf).parent.joinpath('model_fixed.urdf')
        if vhacd_urdf.exists():
            urdf = str(vhacd_urdf)
    else:
        raise ValueError('Unsupported dataset. Must be ShapeNet/PartNet_Mobility')

    loader = env._scene.create_urdf_loader()
    loader.load_multiple_collisions_from_file = True
    loader.scale = articulation_config.get('scale', 1)
    loader.fix_root_link = False

    articulation = loader.load(urdf, config={'material': env.physical_materials['object_material']})
    articulation.set_root_pose(Pose())
    return articulation

def get_art_visual_mesh(art: sapien.Articulation, eps=1e-6) -> o3d.geometry.TriangleMesh:
    """Retrieve visual mesh from the articulation.
        Not using vertex normals, compute the triangle normals with o3d.
    Inputs:
        art: sapien.Articulation
        eps: Parameter that defines the distance between close vertices for merging.
    Output:
        art_mesh: o3d.geometry.TriangleMesh.
    """
    art_mesh = o3d.geometry.TriangleMesh()
    for actor in art.get_links():
        actor_pose = actor.get_pose().to_transformation_matrix()  # T_WA
        for renderbody in actor.get_visual_bodies():
            renderbody_pose = actor_pose @ renderbody.local_pose.to_transformation_matrix()  # T_WB
            for rendershape in renderbody.get_render_shapes():
                m = rendershape.mesh
                vertices = o3d.utility.Vector3dVector(m.vertices)
                triangles = o3d.utility.Vector3iVector(m.indices.reshape(-1,3))
                mesh = o3d.geometry.TriangleMesh(vertices, triangles)
                #mesh.vertex_normals = o3d.utility.Vector3dVector(m.normals)
                np.testing.assert_allclose(renderbody.scale[0], renderbody.scale)
                art_mesh += mesh.scale(renderbody.scale[0], center=np.zeros(3))\
                        .transform(renderbody_pose)
    #art_mesh = art_mesh.normalize_normals().merge_close_vertices(eps)
    art_mesh = art_mesh.merge_close_vertices(eps).compute_triangle_normals()
    return art_mesh


def process_data(obj_model_files, save_aligned_obj_mesh=False, add_table_top_plane=False):
    columns_str_list = ['dataset','instance_id','category',
                        'transform','scale','aligned_dims','aligned_planes']
    target_df = pd.DataFrame([],columns=columns_str_list)

    for obj_cat, file_path in obj_model_files.items():
        if type(file_path) == str:
            with Path(file_path).open('r') as f:
                raw_yaml = yaml.load(f, Loader=yaml.SafeLoader)
            print(f'Loaded "{obj_cat}" models from {file_path}')
        elif type(file_path) == dict:
            raw_yaml = {'obj_0': file_path}
            print(f'Using 1 provided "{obj_cat}" model config')
        elif type(file_path) == list and all([type(i) == dict for i in file_path]):
            raw_yaml = {f'obj_{i}': obj_dict for i, obj_dict in enumerate(file_path)}
            print(f'Using {len(file_path)} provided "{obj_cat}" model configs')
        else:
            raise TypeError(f'Unsupported type of file_path: {type(file_path)}')

        for obj_idx, (k, art_config) in enumerate(raw_yaml.items()):
            if 'partnet_mobility_id' in art_config:
                dataset = "PartNet-Mobility"
                instance_id = art_config['partnet_mobility_id']
            elif 'shapenet_synset_model_id' in art_config:
                dataset = "ShapeNet"
                instance_id = art_config['shapenet_synset_model_id']
            else:
                raise ValueError('Unsupported dataset. Must be ShapeNet/PartNet_Mobility')
            print(f'[ model_preprocessing ] Processing {obj_cat} {obj_idx}/{len(raw_yaml)} id={instance_id}...')
            art_mesh = get_art_visual_mesh(load_articulation(env, art_config))

            scale = art_config['scale']
            aligned_planes, aligned_dims, transform, transform_no_scale \
                = get_aligned_planes(instance_id, art_mesh, scale,
                                     save_mesh=save_aligned_obj_mesh,
                                     add_table_top_plane=(add_table_top_plane and obj_cat == 'Table'))

            aligned_planes_str = '\,'.join([np_array_to_str(x, '\,') for x in aligned_planes])
            aligned_dims_str = np_array_to_str(aligned_dims, '\,')
            transform_str = np_to_str(transform_no_scale, '\,')
            df = pd.DataFrame([[dataset, instance_id, obj_cat, transform_str, scale, aligned_dims_str, aligned_planes_str]],
                              columns=columns_str_list)
            target_df = target_df.append(df)
    return target_df


def get_aligned_planes(instance_id: str, mesh: o3d.geometry.TriangleMesh, scale: float,
                       up=np.array([0,0,1]), front=np.array([0,-1,0]),
                       viz = False, save_mesh = False, add_table_top_plane=False):
    """Input meshes are scaled for ShapeNet/PartNet"""
    # Get aligned coordinate
    if up.size > 0:
        up_axis = np.array(up)
    else:
        up_axis = np.array([0,0,1])
    if front.size > 0:
        front_axis = np.array(front)
    else:
        front_axis = np.array([0,-1,0])

    # Get rotation to canonical coordinate
    RC_B = np.eye(3)
    RC_B[:,1] = -front_axis
    RC_B[:,2] = up_axis
    RC_B[:,0] = np.cross(RC_B[:,1], RC_B[:,2])
    RC_B = np.linalg.inv(RC_B)

    # Transform the mesh: scale(m), axis-aligned
    #mesh = o3d.io.read_triangle_mesh(path_to_dataset + obj_folder + instance_id + '.obj')
    #mesh = o3d.io.read_triangle_mesh(mesh_path)
    pC_B = -(mesh.get_max_bound()+mesh.get_min_bound())/2
    if not np.allclose(0.0, pC_B):
        with np.printoptions(precision=6, suppress=True):
            print(f'[ DEBUG ] Mesh is not centered at [0,0,0]: {pC_B}')

    # mesh.compute_vertex_normals()
    # coor = o3d.geometry.TriangleMesh.create_coordinate_frame()
    # o3d.visualization.draw_geometries([mesh, coor])

    mesh.translate(pC_B)
    mesh.rotate(RC_B, [0,0,0])
    #mesh.scale(scale, [0,0,0])  # Input meshes are scaled for ShapeNet/PartNet

    aligned_dims = mesh.get_max_bound()-mesh.get_min_bound()

    # Record applied transform
    transform = np.eye(4)
    transform[0:3,0:3] = RC_B * scale
    transform[0:3,3] = np.dot(RC_B, pC_B) * scale

    transform_no_scale = np.eye(4)
    transform_no_scale[0:3,0:3] = RC_B
    transform_no_scale[0:3,3] = np.dot(RC_B, pC_B)

    if save_mesh:
        save_mesh_path = output_dataset_folder / f'{instance_id}.obj'
        save_mesh_path.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_triangle_mesh(str(save_mesh_path), mesh)

    # mesh = mesh.simplify_vertex_clustering(voxel_size=0.01, contraction=o3d.geometry.SimplificationContraction.Average)
    #mesh.compute_vertex_normals()
    if viz:
        mesh.paint_uniform_color([0.5, 0.5, 0.5])
        coord = o3d.geometry.TriangleMesh.create_coordinate_frame()
        o3d.visualization.draw_geometries([mesh, coord])
        # mesh_T = copy.deepcopy(mesh).scale(2, mesh.get_center())
        # mesh_T = copy.deepcopy(mesh).transform(T)
        # mesh_T.paint_uniform_color([0.5, 0.5, 0.5])
        # o3d.visualization.draw_geometries([mesh, mesh_T])

    # Sample point cloud from mesh
    #pcd = mesh.sample_points_uniformly(number_of_points=5000)

    # Detect planes
    box_area = np.array([aligned_dims[0]*aligned_dims[1],
                         aligned_dims[1]*aligned_dims[2],
                         aligned_dims[0]*aligned_dims[2]])
    sorted_area = np.sort(box_area)
    min_area = (sorted_area[0]+sorted_area[1]+sorted_area[1])/10
    planes = detect_planes_from_mesh(mesh, distance_threshood=0.02,
                                     angle_threshood=5, max_iterations=800,
                                     min_area=min_area, max_plane_num=15)
    if add_table_top_plane:
        planes = [[0, 0, 1, -mesh.get_max_bound()[-1]]] + planes
        print('Adding table top plane: ', planes[0])
    # planes = []
    print('Num of planes:', len(planes))

    if viz:
        line_set, colors = draw_planes(planes, np.linalg.norm(mesh.get_max_bound()))
        arrow_list = draw_plane_normal(planes, colors)
        o3d.visualization.draw_geometries([pcd]+line_set+arrow_list)

    return planes, aligned_dims, transform, transform_no_scale


if __name__ == "__main__":
    obj_model_files = {
        'Table': [art_dict for art_dict in env.level_config['layout']['articulations'] if art_dict['name'] == 'table'],
        'Bottle': '/home/kolinguo/Desktop/Research/code/mobile_pnp/assets/config_files/models/bottle_models.yml',
        'Bowl': '/home/kolinguo/Desktop/Research/code/mobile_pnp/assets/config_files/models/bowl_models.yml',
        'Can': '/home/kolinguo/Desktop/Research/code/mobile_pnp/assets/config_files/models/can_models.yml',
        'Mug': '/home/kolinguo/Desktop/Research/code/mobile_pnp/assets/config_files/models/mug_models.yml'
    }
    output_dataset_folder.mkdir(parents=True, exist_ok=True)

    target_df = process_data(obj_model_files, save_aligned_obj_mesh=True, add_table_top_plane=True)

    output_csv_path = output_dataset_folder.parent / 'cad_models_partnet_shapenet.csv'
    target_df.to_csv(output_csv_path, sep=',',index=False)
    print(f'Saved preprocessing results to {output_csv_path.resolve()}')
