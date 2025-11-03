"""
Flexible geometric operations node with dynamic operation slots and per-operation outputs.

This module provides a flexible alternative to the fixed-order MeshCraftPostProcess node,
allowing users to configure an arbitrary sequence of geometric operations with intermediate
outputs for debugging and inspection.
"""

from typing import Tuple, Optional
import trimesh
from trimesh import Trimesh

# Import utilities from existing nodes
from .mesh_nodes import reducefacesnano


class MeshCraft_GeomOperations:
    """
    Flexible geometric operations node with configurable operation sequence.

    Features:
    - Dynamic number of operations (1-10) via slider
    - Each operation slot has dropdown to select operation type
    - Operations execute in slot order (top to bottom)
    - Each operation outputs its result for inspection
    - Shared parameters for operation-specific settings

    Available Operations:
    - none: Skip this slot
    - remove_floaters: Remove disconnected mesh components
    - remove_degenerate: Clean invalid faces and vertices
    - reduce_faces: Remesh to target face count
    - smooth_normals: Recalculate vertex normals
    - laplacian_smooth: Geometric surface smoothing
    - ensure_manifold: Fix non-manifold geometry
    """

    # Define available operations
    OPERATIONS = [
        "none",
        "remove_floaters",
        "remove_degenerate",
        "reduce_faces",
        "smooth_normals",
        "laplacian_smooth",
        "ensure_manifold",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "trimesh": ("TRIMESH",),
                "num_operations": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "tooltip": "Number of operations to perform (determines visible slots)"
                }),
            },
            "optional": {}
        }

        # Add 10 operation slots (user sets how many to use via num_operations)
        for i in range(1, 11):
            default_op = "none"
            if i == 1:
                default_op = "remove_floaters"
            elif i == 2:
                default_op = "remove_degenerate"
            elif i == 3:
                default_op = "reduce_faces"

            inputs["optional"][f"operation_{i}"] = (
                cls.OPERATIONS,
                {
                    "default": default_op,
                    "tooltip": f"Operation to perform in slot {i}"
                }
            )

        # Operation-specific parameters (shared across operations)
        inputs["optional"]["max_facenum"] = ("INT", {
            "default": 40000,
            "min": 100,
            "max": 10000000,
            "step": 1000,
            "tooltip": "Target face count for reduce_faces operation"
        })

        inputs["optional"]["smooth_iterations"] = ("INT", {
            "default": 3,
            "min": 1,
            "max": 20,
            "step": 1,
            "tooltip": "Number of iterations for laplacian_smooth"
        })

        inputs["optional"]["lambda_factor"] = ("FLOAT", {
            "default": 0.5,
            "min": 0.0,
            "max": 1.0,
            "step": 0.05,
            "tooltip": "Smoothing strength for laplacian_smooth (0=none, 1=max)"
        })

        return inputs

    # Return 10 mesh outputs (one per operation slot)
    RETURN_TYPES = ("TRIMESH",) * 10
    RETURN_NAMES = tuple(f"mesh_after_op{i}" for i in range(1, 11))
    FUNCTION = "process"
    CATEGORY = "MeshCraft/Operations"

    def process(
        self,
        trimesh: Trimesh,
        num_operations: int,
        **kwargs
    ) -> Tuple[Optional[Trimesh], ...]:
        """
        Execute geometric operations in sequence, outputting intermediate results.

        Args:
            trimesh: Input mesh
            num_operations: Number of operations to perform
            **kwargs: Operation parameters and operation slot selections

        Returns:
            Tuple of 10 meshes (one per operation slot, None for unused slots)
        """
        # Initialize outputs
        outputs = [None] * 10
        current_mesh = trimesh.copy()

        print(f"\n{'='*60}")
        print(f"MeshCraft GeomOperations: {num_operations} operations")
        print(f"{'='*60}")

        # Execute operations in order
        for i in range(1, num_operations + 1):
            op_name = kwargs.get(f"operation_{i}", "none")

            if op_name == "none":
                print(f"  Slot {i}: Skipped")
                outputs[i-1] = current_mesh.copy()
                continue

            try:
                # Apply operation
                current_mesh = self._apply_operation(current_mesh, op_name, kwargs)
                outputs[i-1] = current_mesh.copy()

                # Print stats
                print(f"  Slot {i}: {op_name} → {len(current_mesh.vertices)} verts, "
                      f"{len(current_mesh.faces)} faces")

            except Exception as e:
                print(f"  ⚠️  Slot {i}: {op_name} FAILED - {e}")
                # On error, keep previous mesh
                outputs[i-1] = current_mesh.copy()

        # Fill remaining unused slots with the final mesh
        for i in range(num_operations, 10):
            outputs[i] = current_mesh.copy()

        print(f"{'='*60}\n")

        return tuple(outputs)

    def _apply_operation(
        self,
        mesh: Trimesh,
        operation: str,
        params: dict
    ) -> Trimesh:
        """
        Apply a single geometric operation to the mesh.

        Args:
            mesh: Input mesh
            operation: Operation name
            params: Operation parameters

        Returns:
            Processed mesh
        """
        new_mesh = mesh.copy()

        if operation == "remove_floaters":
            new_mesh = self._remove_floaters(new_mesh)

        elif operation == "remove_degenerate":
            new_mesh = self._remove_degenerate(new_mesh)

        elif operation == "reduce_faces":
            max_facenum = params.get("max_facenum", 40000)
            new_mesh = self._reduce_faces(new_mesh, max_facenum)

        elif operation == "smooth_normals":
            new_mesh = self._smooth_normals(new_mesh)

        elif operation == "laplacian_smooth":
            iterations = params.get("smooth_iterations", 3)
            lambda_factor = params.get("lambda_factor", 0.5)
            new_mesh = self._laplacian_smooth(new_mesh, iterations, lambda_factor)

        elif operation == "ensure_manifold":
            new_mesh = self._ensure_manifold(new_mesh)

        else:
            print(f"Unknown operation: {operation}")

        return new_mesh

    # =====================================================================
    # Individual Operation Implementations
    # =====================================================================

    def _remove_floaters(self, mesh: Trimesh) -> Trimesh:
        """Remove disconnected geometry, keeping only the largest component."""
        components = mesh.split(only_watertight=False)
        if len(components) == 0:
            return mesh

        # Find largest component by face count
        largest = max(components, key=lambda c: len(c.faces))
        return largest

    def _remove_degenerate(self, mesh: Trimesh) -> Trimesh:
        """Remove degenerate faces and clean up mesh."""
        mesh.remove_degenerate_faces()
        mesh.remove_duplicate_faces()
        mesh.remove_infinite_values()
        mesh.remove_unreferenced_vertices()
        return mesh

    def _reduce_faces(self, mesh: Trimesh, max_facenum: int) -> Trimesh:
        """Reduce face count using Instant Meshes remeshing."""
        if len(mesh.faces) <= max_facenum:
            print(f"    (Already below target: {len(mesh.faces)} ≤ {max_facenum})")
            return mesh

        return reducefacesnano(mesh, max_facenum)

    def _smooth_normals(self, mesh: Trimesh) -> Trimesh:
        """Recalculate vertex normals for smooth shading."""
        mesh.vertex_normals = trimesh.smoothing.get_vertices_normals(mesh)
        return mesh

    def _laplacian_smooth(
        self,
        mesh: Trimesh,
        iterations: int,
        lambda_factor: float
    ) -> Trimesh:
        """
        Apply Laplacian smoothing to mesh geometry.

        Args:
            mesh: Input mesh
            iterations: Number of smoothing iterations
            lambda_factor: Smoothing strength (0-1)
        """
        return trimesh.smoothing.filter_laplacian(
            mesh,
            iterations=iterations,
            lamb=lambda_factor
        )

    def _ensure_manifold(self, mesh: Trimesh) -> Trimesh:
        """
        Attempt to fix non-manifold geometry.

        This is a best-effort operation that may not always succeed.
        """
        # Fill holes (simple approach)
        if hasattr(mesh, 'fill_holes'):
            mesh.fill_holes()

        # Remove duplicate vertices with tolerance
        mesh.merge_vertices()

        # Remove degenerate faces
        mesh.remove_degenerate_faces()

        # Check if manifold
        if not mesh.is_watertight:
            print(f"    (Warning: Mesh is still not watertight)")

        return mesh


# Node registration
NODE_CLASS_MAPPINGS = {
    "MeshCraft_GeomOperations": MeshCraft_GeomOperations,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MeshCraft_GeomOperations": "MeshCraft Geometric Operations",
}
