"""
Individual geometric operation nodes for MeshCraft.

This module provides focused, single-purpose nodes for mesh operations,
replacing the multi-operation GeomOperations node for better visual clarity
and workflow flexibility in ComfyUI.
"""

import trimesh
from trimesh import Trimesh

# Import utilities from existing nodes
from .mesh_nodes import reducefacesnano


# ============================================================================
# Cleanup Operations
# ============================================================================

class MeshCraft_RemoveFloaters:
    """
    Remove disconnected mesh components, keeping only the largest.

    Useful for cleaning up floating geometry, isolated vertices,
    or mesh debris after boolean operations.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH", {"tooltip": "Input mesh to clean"}),
            }
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("trimesh",)
    FUNCTION = "process"
    CATEGORY = "MeshCraft/Operations"

    def process(self, trimesh: Trimesh) -> tuple:
        """Remove disconnected components, keeping largest."""
        mesh = trimesh.copy()

        components = mesh.split(only_watertight=False)
        if len(components) == 0:
            return (mesh,)

        # Find largest component by face count
        largest = max(components, key=lambda c: len(c.faces))

        print(f"MeshCraft Remove Floaters: {len(components)} components → kept largest "
              f"({len(largest.vertices)} verts, {len(largest.faces)} faces)")

        return (largest,)


class MeshCraft_RemoveDegenerate:
    """
    Clean invalid mesh geometry including degenerate faces, duplicates,
    infinite values, and unreferenced vertices.

    Essential for fixing corrupted meshes or preparing for export.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH", {"tooltip": "Input mesh to clean"}),
            }
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("trimesh",)
    FUNCTION = "process"
    CATEGORY = "MeshCraft/Operations"

    def process(self, trimesh: Trimesh) -> tuple:
        """Remove degenerate faces and clean up mesh."""
        mesh = trimesh.copy()

        initial_verts = len(mesh.vertices)
        initial_faces = len(mesh.faces)

        mesh.remove_degenerate_faces()
        mesh.remove_duplicate_faces()
        mesh.remove_infinite_values()
        mesh.remove_unreferenced_vertices()

        print(f"MeshCraft Remove Degenerate: {initial_verts} → {len(mesh.vertices)} verts, "
              f"{initial_faces} → {len(mesh.faces)} faces")

        return (mesh,)


# ============================================================================
# Optimization Operations
# ============================================================================

class MeshCraft_ReduceFaces:
    """
    Reduce polygon count through remeshing using Instant Meshes algorithm.

    Optimizes mesh for performance, reduces file size, while attempting
    to preserve overall shape and features.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH", {"tooltip": "Input mesh to optimize"}),
                "max_facenum": ("INT", {
                    "default": 40000,
                    "min": 100,
                    "max": 10000000,
                    "step": 1000,
                    "tooltip": "Target face count (actual result may be ~2x due to algorithm)"
                }),
            }
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("trimesh",)
    FUNCTION = "process"
    CATEGORY = "MeshCraft/Operations"

    def process(self, trimesh: Trimesh, max_facenum: int) -> tuple:
        """Reduce face count using Instant Meshes remeshing."""
        mesh = trimesh.copy()

        if len(mesh.faces) <= max_facenum:
            print(f"MeshCraft Reduce Faces: Already below target "
                  f"({len(mesh.faces)} ≤ {max_facenum})")
            return (mesh,)

        # reducefacesnano outputs ~2x the requested face count, so request half
        result = reducefacesnano(mesh, max_facenum // 2)

        print(f"MeshCraft Reduce Faces: {len(mesh.faces)} → {len(result.faces)} faces "
              f"(target: {max_facenum})")

        return (result,)


# ============================================================================
# Appearance Operations
# ============================================================================

class MeshCraft_SmoothNormals:
    """
    Recalculate vertex normals for smooth shading.

    Improves visual appearance without changing geometry.
    Essential for meshes with faceted shading that should appear smooth.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH", {"tooltip": "Input mesh to smooth"}),
            }
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("trimesh",)
    FUNCTION = "process"
    CATEGORY = "MeshCraft/Operations"

    def process(self, trimesh: Trimesh) -> tuple:
        """Recalculate vertex normals for smooth shading."""
        mesh = trimesh.copy()

        mesh.vertex_normals = trimesh.smoothing.get_vertices_normals(mesh)

        print(f"MeshCraft Smooth Normals: Recalculated normals for "
              f"{len(mesh.vertices)} vertices")

        return (mesh,)


# ============================================================================
# Geometry Operations
# ============================================================================

class MeshCraft_LaplacianSmooth:
    """
    Apply Laplacian smoothing to mesh geometry.

    Actually moves vertices to smooth out surface bumps and irregularities.
    Unlike normal smoothing, this changes the mesh geometry.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH", {"tooltip": "Input mesh to smooth"}),
                "iterations": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 20,
                    "step": 1,
                    "tooltip": "Number of smoothing iterations (more = smoother)"
                }),
                "lambda_factor": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Smoothing strength (0=none, 1=maximum)"
                }),
            }
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("trimesh",)
    FUNCTION = "process"
    CATEGORY = "MeshCraft/Operations"

    def process(self, trimesh: Trimesh, iterations: int, lambda_factor: float) -> tuple:
        """Apply Laplacian smoothing to mesh geometry."""
        mesh = trimesh.copy()

        result = trimesh.smoothing.filter_laplacian(
            mesh,
            iterations=iterations,
            lamb=lambda_factor
        )

        print(f"MeshCraft Laplacian Smooth: Applied {iterations} iterations "
              f"(λ={lambda_factor:.2f}) to {len(mesh.vertices)} vertices")

        return (result,)


# ============================================================================
# Repair Operations
# ============================================================================

class MeshCraft_EnsureManifold:
    """
    Attempt to fix non-manifold geometry.

    Fills holes, merges duplicate vertices, removes degenerate faces.
    This is a best-effort operation that may not always succeed.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH", {"tooltip": "Input mesh to repair"}),
            }
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("trimesh",)
    FUNCTION = "process"
    CATEGORY = "MeshCraft/Operations"

    def process(self, trimesh: Trimesh) -> tuple:
        """Attempt to fix non-manifold geometry."""
        mesh = trimesh.copy()

        initial_watertight = mesh.is_watertight

        # Fill holes (simple approach)
        if hasattr(mesh, 'fill_holes'):
            mesh.fill_holes()

        # Remove duplicate vertices with tolerance
        mesh.merge_vertices()

        # Remove degenerate faces
        mesh.remove_degenerate_faces()

        final_watertight = mesh.is_watertight

        if not initial_watertight and final_watertight:
            print(f"MeshCraft Ensure Manifold: ✓ Successfully made mesh watertight")
        elif not final_watertight:
            print(f"MeshCraft Ensure Manifold: ⚠ Mesh is still not watertight "
                  f"({len(mesh.vertices)} verts, {len(mesh.faces)} faces)")
        else:
            print(f"MeshCraft Ensure Manifold: Mesh was already watertight")

        return (mesh,)


# ============================================================================
# Node Registration
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "MeshCraft_RemoveFloaters": MeshCraft_RemoveFloaters,
    "MeshCraft_RemoveDegenerate": MeshCraft_RemoveDegenerate,
    "MeshCraft_ReduceFaces": MeshCraft_ReduceFaces,
    "MeshCraft_SmoothNormals": MeshCraft_SmoothNormals,
    "MeshCraft_LaplacianSmooth": MeshCraft_LaplacianSmooth,
    "MeshCraft_EnsureManifold": MeshCraft_EnsureManifold,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MeshCraft_RemoveFloaters": "Remove Floaters",
    "MeshCraft_RemoveDegenerate": "Remove Degenerate Faces",
    "MeshCraft_ReduceFaces": "Reduce Faces",
    "MeshCraft_SmoothNormals": "Smooth Normals",
    "MeshCraft_LaplacianSmooth": "Laplacian Smooth",
    "MeshCraft_EnsureManifold": "Ensure Manifold",
}
