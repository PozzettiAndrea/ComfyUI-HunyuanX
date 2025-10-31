# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import os
import tempfile
from typing import Union

import numpy as np
import pymeshlab
import torch
import trimesh

from .models.autoencoders import Latent2MeshOutput
from .utils import synchronize_timer


def load_mesh(path):
    if path.endswith(".glb"):
        mesh = trimesh.load(path)
    else:
        mesh = pymeshlab.MeshSet()
        mesh.load_new_mesh(path)
    return mesh


def reduce_face(mesh: pymeshlab.MeshSet, max_facenum: int = 200000):
    if max_facenum > mesh.current_mesh().face_number():
        return mesh

    mesh.apply_filter(
        "meshing_decimation_quadric_edge_collapse",
        targetfacenum=max_facenum,
        qualitythr=1.0,
        preserveboundary=True,
        boundaryweight=3,
        preservenormal=True,
        preservetopology=True,
        autoclean=True
    )
    return mesh


def remove_floater(mesh: pymeshlab.MeshSet):
    mesh.apply_filter("compute_selection_by_small_disconnected_components_per_face",
                      nbfaceratio=0.005)
    mesh.apply_filter("compute_selection_transfer_face_to_vertex", inclusive=False)
    mesh.apply_filter("meshing_remove_selected_vertices_and_faces")
    return mesh


def pymeshlab2trimesh(mesh: pymeshlab.MeshSet):
    # Direct conversion without file I/O - avoids pymeshlab OpenGL dependency issues
    # Extract vertex and face arrays directly from pymeshlab
    current_mesh = mesh.current_mesh()
    vertices = current_mesh.vertex_matrix()
    faces = current_mesh.face_matrix()

    # Create trimesh directly from arrays
    return trimesh.Trimesh(vertices=vertices, faces=faces)


def trimesh2pymeshlab(mesh: trimesh.Trimesh):
    # Handle scene by merging all geometries
    if isinstance(mesh, trimesh.scene.Scene):
        combined_mesh = None
        for obj in mesh.geometry.values():
            if combined_mesh is None:
                combined_mesh = obj
            else:
                combined_mesh = combined_mesh + obj
        mesh = combined_mesh

    # Direct conversion without file I/O - avoids pymeshlab OpenGL dependency issues
    # Create pymeshlab Mesh directly from vertex and face arrays
    import numpy as np
    vertex_matrix = np.array(mesh.vertices, dtype=np.float64)
    face_matrix = np.array(mesh.faces, dtype=np.int32)

    ms = pymeshlab.MeshSet()
    mesh_pymeshlab = pymeshlab.Mesh(vertex_matrix=vertex_matrix, face_matrix=face_matrix)
    ms.add_mesh(mesh_pymeshlab, "converted_mesh")
    return ms


def export_mesh(input, output):
    if isinstance(input, pymeshlab.MeshSet):
        mesh = output
    elif isinstance(input, Latent2MeshOutput):
        output = Latent2MeshOutput()
        output.mesh_v = output.current_mesh().vertex_matrix()
        output.mesh_f = output.current_mesh().face_matrix()
        mesh = output
    else:
        mesh = pymeshlab2trimesh(output)
    return mesh


def import_mesh(mesh: Union[pymeshlab.MeshSet, trimesh.Trimesh, Latent2MeshOutput, str]) -> pymeshlab.MeshSet:
    if isinstance(mesh, str):
        mesh = load_mesh(mesh)
    elif isinstance(mesh, Latent2MeshOutput):
        mesh = pymeshlab.MeshSet()
        mesh_pymeshlab = pymeshlab.Mesh(vertex_matrix=mesh.mesh_v, face_matrix=mesh.mesh_f)
        mesh.add_mesh(mesh_pymeshlab, "converted_mesh")

    if isinstance(mesh, (trimesh.Trimesh, trimesh.scene.Scene)):
        mesh = trimesh2pymeshlab(mesh)

    return mesh


class FaceReducer:
    @synchronize_timer('FaceReducer')
    def __call__(
        self,
        mesh: Union[pymeshlab.MeshSet, trimesh.Trimesh, Latent2MeshOutput, str],
        max_facenum: int = 40000
    ) -> Union[pymeshlab.MeshSet, trimesh.Trimesh]:
        ms = import_mesh(mesh)
        ms = reduce_face(ms, max_facenum=max_facenum)
        mesh = export_mesh(mesh, ms)
        return mesh


class FloaterRemover:
    @synchronize_timer('FloaterRemover')
    def __call__(
        self,
        mesh: Union[pymeshlab.MeshSet, trimesh.Trimesh, Latent2MeshOutput, str],
    ) -> Union[pymeshlab.MeshSet, trimesh.Trimesh, Latent2MeshOutput]:
        ms = import_mesh(mesh)
        ms = remove_floater(ms)
        mesh = export_mesh(mesh, ms)
        return mesh


class DegenerateFaceRemover:
    @synchronize_timer('DegenerateFaceRemover')
    def __call__(
        self,
        mesh: Union[pymeshlab.MeshSet, trimesh.Trimesh, Latent2MeshOutput, str],
    ) -> Union[pymeshlab.MeshSet, trimesh.Trimesh, Latent2MeshOutput]:
        ms = import_mesh(mesh)

        # Actually remove degenerate faces using proper pymeshlab filters
        # No file I/O needed - avoids OpenGL dependency issues
        ms.apply_filter('meshing_remove_null_faces')  # Remove 0-area faces
        ms.apply_filter('meshing_remove_duplicate_faces')  # Remove duplicate faces
        ms.apply_filter('meshing_remove_unreferenced_vertices')  # Clean up unused vertices

        mesh = export_mesh(mesh, ms)
        return mesh


def mesh_normalize(mesh):
    """
    Normalize mesh vertices to sphere
    """
    scale_factor = 1.2
    vtx_pos = np.asarray(mesh.vertices)
    max_bb = (vtx_pos - 0).max(0)[0]
    min_bb = (vtx_pos - 0).min(0)[0]

    center = (max_bb + min_bb) / 2

    scale = torch.norm(torch.tensor(vtx_pos - center, dtype=torch.float32), dim=1).max() * 2.0

    vtx_pos = (vtx_pos - center) * (scale_factor / float(scale))
    mesh.vertices = vtx_pos

    return mesh


class MeshSimplifier:
    def __init__(self, executable: str = None):
        if executable is None:
            CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
            executable = os.path.join(CURRENT_DIR, "mesh_simplifier.bin")
        self.executable = executable

    @synchronize_timer('MeshSimplifier')
    def __call__(
        self,
        mesh: Union[trimesh.Trimesh],
    ) -> Union[trimesh.Trimesh]:
        with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as temp_input:
            with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as temp_output:
                mesh.export(temp_input.name)
                os.system(f'{self.executable} {temp_input.name} {temp_output.name}')
                ms = trimesh.load(temp_output.name, process=False)
                if isinstance(ms, trimesh.Scene):
                    combined_mesh = trimesh.Trimesh()
                    for geom in ms.geometry.values():
                        combined_mesh = trimesh.util.concatenate([combined_mesh, geom])
                    ms = combined_mesh
                ms = mesh_normalize(ms)
                return ms
