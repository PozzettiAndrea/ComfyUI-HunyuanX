"""
ComfyUI-MeshCraft Nodes
Advanced mesh manipulation and optimization nodes for ComfyUI
"""

import torch
import trimesh
import numpy as np
from typing import Tuple


def reducefacesnano(new_mesh, max_facenum: int):
    """
    Reduce mesh face count using Instant Meshes (pynanoinstantmeshes).

    Args:
        new_mesh: trimesh.Trimesh object
        max_facenum: Target maximum number of faces

    Returns:
        trimesh.Trimesh with reduced face count
    """
    try:
        import pynanoinstantmeshes as PyNIM

        current_faces = len(new_mesh.faces)

        # Target vertex count (Instant Meshes works with vertices)
        target_vertices = max(100, int(max_facenum * 0.25))

        print(f"Remeshing from {current_faces} faces to ~{max_facenum} target faces...")
        print(f"Requesting {target_vertices} vertices from Instant Meshes...")

        # Remesh with Instant Meshes
        new_verts, new_faces = PyNIM.remesh(
            np.array(new_mesh.vertices, dtype=np.float32),
            np.array(new_mesh.faces, dtype=np.uint32),
            target_vertices,
            align_to_boundaries=True,
            smooth_iter=2
        )

        # Instant Meshes can fail, check validity
        if new_verts.shape[0] - 1 != new_faces.max():
            raise ValueError("Remeshing failed - invalid vertex indices")

        # Triangulate quads (Instant Meshes outputs quads)
        new_faces = trimesh.geometry.triangulate_quads(new_faces)

        new_mesh = trimesh.Trimesh(
            vertices=new_verts.astype(np.float32),
            faces=new_faces
        )

        print(f"Remeshed: {new_mesh.vertices.shape[0]} vertices, {new_mesh.faces.shape[0]} faces")
        return new_mesh

    except Exception as e:
        print(f"Instant Meshes failed: {e}, returning original mesh")
        return new_mesh


class MeshCraftPostProcess:
    """
    Advanced mesh post-processing node for ComfyUI.

    Provides mesh cleaning, optimization, and quality improvements:
    - Remove disconnected floater geometry
    - Remove degenerate faces (zero area, duplicate vertices)
    - Reduce face count using Instant Meshes
    - Smooth vertex normals
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "remove_floaters": ("BOOLEAN", {"default": True}),
                "remove_degenerate_faces": ("BOOLEAN", {"default": True}),
                "reduce_faces": ("BOOLEAN", {"default": True}),
                "max_facenum": ("INT", {
                    "default": 40000,
                    "min": 1,
                    "max": 10000000,
                    "step": 1
                }),
                "smooth_normals": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("trimesh",)
    FUNCTION = "process"
    CATEGORY = "MeshCraft"

    def process(
        self,
        trimesh: trimesh.Trimesh,
        remove_floaters: bool,
        remove_degenerate_faces: bool,
        reduce_faces: bool,
        max_facenum: int,
        smooth_normals: bool
    ) -> Tuple[trimesh.Trimesh]:
        """
        Process mesh with selected optimizations.

        Args:
            trimesh: Input mesh
            remove_floaters: Remove disconnected geometry
            remove_degenerate_faces: Remove invalid faces
            reduce_faces: Apply face reduction
            max_facenum: Target face count
            smooth_normals: Smooth vertex normals

        Returns:
            Tuple containing processed mesh
        """
        new_mesh = trimesh.copy()

        if remove_floaters:
            # Split mesh into connected components and keep only the largest
            components = new_mesh.split(only_watertight=False)
            if len(components) > 0:
                new_mesh = components[0]  # Largest component by default
                for component in components[1:]:
                    if len(component.faces) > len(new_mesh.faces):
                        new_mesh = component
            print(f"Removed floaters: {new_mesh.vertices.shape[0]} vertices, "
                  f"{new_mesh.faces.shape[0]} faces")

        if remove_degenerate_faces:
            # Remove degenerate faces (zero area, duplicate vertices, etc)
            new_mesh.remove_degenerate_faces()
            new_mesh.remove_duplicate_faces()
            new_mesh.remove_infinite_values()
            print(f"Removed degenerate faces: {new_mesh.vertices.shape[0]} vertices, "
                  f"{new_mesh.faces.shape[0]} faces")

        if reduce_faces:
            # Simplify mesh using Instant Meshes
            new_mesh = reducefacesnano(new_mesh, max_facenum)
            print(f"Reduced faces: {new_mesh.vertices.shape[0]} vertices, "
                  f"{new_mesh.faces.shape[0]} faces")

        if smooth_normals:
            # Smooth vertex normals
            new_mesh.vertex_normals = trimesh.smoothing.get_vertices_normals(new_mesh)
            print("Smoothed vertex normals")

        return (new_mesh,)


# Required exports for ComfyUI
NODE_CLASS_MAPPINGS = {
    "MeshCraftPostProcess": MeshCraftPostProcess,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MeshCraftPostProcess": "MeshCraft Post-Process",
}
