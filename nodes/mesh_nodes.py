"""
ComfyUI-MeshCraft Nodes
Advanced mesh manipulation and optimization nodes for ComfyUI
"""

import torch
import trimesh
import numpy as np
import io
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
        # Fallback to trimesh's quadric edge collapse decimation
        print(f"Instant Meshes not available ({e}), using trimesh quadric decimation fallback...")

        try:
            # Use trimesh's simplify_quadric_decimation (requires fast-simplification package)
            decimated_mesh = new_mesh.simplify_quadric_decimation(max_facenum)

            print(f"Decimated: {decimated_mesh.vertices.shape[0]} vertices, {decimated_mesh.faces.shape[0]} faces")
            return decimated_mesh

        except ImportError as import_error:
            print(f"fast-simplification not installed ({import_error})")
            print("Install with: pip install fast-simplification")
            print("Falling back to vertex clustering...")

            try:
                # Fallback #2: vertex clustering (lower quality but no dependencies)
                bounds_size = np.linalg.norm(new_mesh.bounds[1] - new_mesh.bounds[0])
                target_edge_length = bounds_size * np.sqrt(1.0 / max_facenum) * 2.0
                clustered_mesh = new_mesh.simplify_vertex_clustering(tolerance=target_edge_length)

                print(f"Clustered: {clustered_mesh.vertices.shape[0]} vertices, {clustered_mesh.faces.shape[0]} faces")
                return clustered_mesh

            except Exception as cluster_error:
                print(f"All fallbacks failed: {cluster_error}, returning original mesh")
                return new_mesh

        except Exception as fallback_error:
            print(f"Decimation failed: {fallback_error}, returning original mesh")
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


class RenderMeshMultiView:
    """
    Render mesh from 6 orthographic viewpoints matching Hunyuan3D 2.1 camera configuration.

    Outputs 6 views: Front, Right, Back, Left, Top, Bottom
    Camera poses match Hunyuan3D exactly for use in texture generation/editing workflows.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("TRIMESH",),
                "resolution": ("INT", {
                    "default": 512,
                    "min": 128,
                    "max": 2048,
                    "step": 64,
                    "tooltip": "Render resolution for each view"
                }),
                "camera_distance": ("FLOAT", {
                    "default": 1.5,
                    "min": 0.5,
                    "max": 5.0,
                    "step": 0.1,
                    "tooltip": "Camera distance from object center (Hunyuan3D default: 1.5)"
                }),
                "output_layout": (["grid_2x3", "grid_3x2", "horizontal_1x6", "batch"], {
                    "default": "grid_2x3",
                    "tooltip": "How to arrange the 6 views"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("rendered_views",)
    FUNCTION = "render"
    CATEGORY = "MeshCraft/Rendering"

    def render(self, mesh, resolution, camera_distance, output_layout):
        """
        Render mesh from 6 orthographic viewpoints.

        Args:
            mesh: Input mesh (can be textured or untextured)
            resolution: Render resolution per view
            camera_distance: Distance from camera to object
            output_layout: How to arrange output views

        Returns:
            Tuple containing rendered image(s)
        """
        print(f"\n🎥 Rendering mesh from 6 orthographic views...")
        print(f"   Resolution: {resolution}x{resolution}")
        print(f"   Camera distance: {camera_distance}")
        print(f"   Layout: {output_layout}")

        # Hunyuan3D 2.1 camera configuration (6 orthographic views)
        # Format: (azimuth_degrees, elevation_degrees, description)
        camera_poses = [
            (0, 0, "Front"),      # View 1
            (90, 0, "Right"),     # View 2
            (180, 0, "Back"),     # View 3
            (270, 0, "Left"),     # View 4
            (0, 90, "Top"),       # View 5
            (180, -90, "Bottom"), # View 6
        ]

        # Import PIL for image handling
        try:
            from PIL import Image
        except ImportError:
            raise ImportError("PIL/Pillow is required for rendering. Install with: pip install Pillow")

        # Render each view
        rendered_images = []

        for azimuth, elevation, description in camera_poses:
            print(f"   Rendering {description} view (azimuth={azimuth}°, elevation={elevation}°)...")

            # Create scene with mesh
            scene = trimesh.scene.Scene(mesh)

            # Set orthographic camera
            # Convert angles to radians
            azimuth_rad = np.radians(azimuth)
            elevation_rad = np.radians(elevation)

            # Calculate camera position (spherical coordinates)
            x = camera_distance * np.cos(elevation_rad) * np.sin(azimuth_rad)
            y = camera_distance * np.sin(elevation_rad)
            z = camera_distance * np.cos(elevation_rad) * np.cos(azimuth_rad)

            camera_position = np.array([x, y, z])

            # Set camera transform (look at origin)
            scene.camera_transform = scene.camera.look_at(
                points=[mesh.bounds.mean(axis=0)],
                center=camera_position
            )

            # Render with orthographic projection
            # Use OSMesa for headless rendering (no display required)
            import os
            os.environ['PYOPENGL_PLATFORM'] = 'osmesa'

            try:
                # Try pyrender backend (better quality)
                rendered = scene.save_image(resolution=(resolution, resolution), visible=False)
                img = Image.open(io.BytesIO(rendered))
            except Exception as e:
                # Fallback to simple GL rendering
                print(f"Render attempt 1 failed: {e}, trying fallback...")
                import io
                rendered = scene.save_image(resolution=(resolution, resolution))
                img = Image.open(io.BytesIO(rendered))

            rendered_images.append(np.array(img))

        # Convert to torch tensors (ComfyUI IMAGE format: NHWC, values 0-1)
        rendered_tensors = [torch.from_numpy(img.astype(np.float32) / 255.0) for img in rendered_images]

        # Arrange according to output_layout
        if output_layout == "batch":
            # Stack as batch: (6, H, W, C)
            output = torch.stack(rendered_tensors, dim=0)

        elif output_layout == "horizontal_1x6":
            # Horizontal strip: (1, H, W*6, C)
            output = torch.cat(rendered_tensors, dim=1).unsqueeze(0)

        elif output_layout == "grid_2x3":
            # 2 rows × 3 columns
            # Row 1: Front, Right, Back
            # Row 2: Left, Top, Bottom
            row1 = torch.cat(rendered_tensors[0:3], dim=1)  # Front, Right, Back
            row2 = torch.cat(rendered_tensors[3:6], dim=1)  # Left, Top, Bottom
            output = torch.cat([row1, row2], dim=0).unsqueeze(0)

        elif output_layout == "grid_3x2":
            # 3 rows × 2 columns
            # Row 1: Front, Right
            # Row 2: Back, Left
            # Row 3: Top, Bottom
            row1 = torch.cat([rendered_tensors[0], rendered_tensors[1]], dim=1)
            row2 = torch.cat([rendered_tensors[2], rendered_tensors[3]], dim=1)
            row3 = torch.cat([rendered_tensors[4], rendered_tensors[5]], dim=1)
            output = torch.cat([row1, row2, row3], dim=0).unsqueeze(0)

        print(f"   ✅ Rendered 6 views, output shape: {output.shape}")
        print("=== End Multi-View Rendering ===\n")

        return (output,)


class MeshCraftUVUnwrap:
    """
    UV unwrapping node with multiple algorithm choices.

    Supports two unwrapping methods:
    - xatlas: Fast, automatic UV parameterization (no external dependencies)
    - blender: Blender's Smart UV Project (requires Blender installed)

    XAtlas is faster and works on all systems, while Blender offers more control
    over seam placement and island margins.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "method": (["xatlas", "blender"], {
                    "default": "xatlas",
                    "tooltip": "UV unwrapping algorithm (xatlas is faster, blender offers more control)"
                }),
            },
            "optional": {
                # Blender-specific parameters
                "angle_limit": ("FLOAT", {
                    "default": 66.0,
                    "min": 0.0,
                    "max": 180.0,
                    "step": 0.1,
                    "tooltip": "[Blender only] Angle threshold for creating seams (degrees)"
                }),
                "island_margin": ("FLOAT", {
                    "default": 0.02,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.001,
                    "tooltip": "[Blender only] Spacing between UV islands (0-1)"
                }),
            }
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("trimesh",)
    FUNCTION = "unwrap"
    CATEGORY = "MeshCraft"

    def unwrap(self, trimesh: trimesh.Trimesh, method: str = "xatlas",
               angle_limit: float = 66.0, island_margin: float = 0.02):
        """
        Unwrap mesh UVs using the selected method.

        Args:
            trimesh: Input mesh (trimesh.Trimesh or Scene)
            method: "xatlas" or "blender"
            angle_limit: [Blender only] Angle for seam detection (degrees)
            island_margin: [Blender only] UV island spacing (0-1)

        Returns:
            Mesh with UV coordinates
        """
        import time

        # Handle Scene objects
        if isinstance(trimesh, trimesh.Scene):
            trimesh = trimesh.dump(concatenate=True)

        print(f"\n{'='*60}")
        print(f"UV Unwrapping with {method.upper()}")
        print(f"Input: {len(trimesh.vertices)} vertices, {len(trimesh.faces)} faces")
        print(f"{'='*60}")

        start_time = time.time()

        if method == "xatlas":
            unwrapped = self._unwrap_xatlas(trimesh)
        elif method == "blender":
            unwrapped = self._unwrap_blender(trimesh, angle_limit, island_margin)
        else:
            raise ValueError(f"Unknown UV unwrapping method: {method}")

        elapsed = time.time() - start_time

        print(f"\n✅ UV unwrapping complete!")
        print(f"   Method: {method}")
        print(f"   Time: {elapsed:.2f}s")
        print(f"   Output: {len(unwrapped.vertices)} vertices, {len(unwrapped.faces)} faces")
        print(f"   UVs: {'Yes' if unwrapped.visual.uv is not None else 'No'}")
        print(f"{'='*60}\n")

        return (unwrapped,)

    def _unwrap_xatlas(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """Unwrap using xatlas (fast, no external dependencies)"""
        import xatlas

        vertices = mesh.vertices
        faces = mesh.faces

        print("Running xatlas parametrization...")

        # Run xatlas
        vmapping, indices, uvs = xatlas.parametrize(vertices, faces)

        # Remap vertices and faces
        new_vertices = vertices[vmapping]
        new_faces = indices

        # Create new mesh with UVs
        unwrapped = trimesh.Trimesh(
            vertices=new_vertices,
            faces=new_faces,
            process=False
        )

        # Set UV coordinates
        unwrapped.visual = trimesh.visual.TextureVisuals(uv=uvs)

        return unwrapped

    def _unwrap_blender(self, mesh: trimesh.Trimesh, angle_limit: float,
                       island_margin: float) -> trimesh.Trimesh:
        """Unwrap using Blender's Smart UV Project"""
        import subprocess
        import tempfile
        import os

        print(f"Running Blender Smart UV Project...")
        print(f"   Angle limit: {angle_limit}°")
        print(f"   Island margin: {island_margin}")

        # Create temp files
        with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as f_in:
            input_path = f_in.name
            mesh.export(input_path)

        with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as f_out:
            output_path = f_out.name

        try:
            # Blender script for UV unwrapping
            script = f"""
import bpy
import numpy as np

# Clear scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Import mesh (OBJ preserves geometry)
bpy.ops.wm.obj_import(filepath='{input_path}')

# Get imported object
obj = bpy.context.selected_objects[0]
bpy.context.view_layer.objects.active = obj

# Switch to edit mode and unwrap
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.uv.smart_project(
    angle_limit={np.radians(angle_limit)},
    island_margin={island_margin},
    area_weight=0.0,
    correct_aspect=True,
    scale_to_bounds=False
)
bpy.ops.object.mode_set(mode='OBJECT')

# Export with UVs
bpy.ops.wm.obj_export(
    filepath='{output_path}',
    export_selected_objects=True,
    export_uv=True,
    export_materials=False
)
"""

            result = subprocess.run([
                'blender', '--background', '--python-expr', script
            ], capture_output=True, text=True, timeout=300)

            if result.returncode != 0:
                raise RuntimeError(f"Blender failed: {result.stderr}")

            # Load the unwrapped mesh
            unwrapped = trimesh.load(output_path, process=False)

            return unwrapped

        finally:
            # Cleanup temp files
            if os.path.exists(input_path):
                os.unlink(input_path)
            if os.path.exists(output_path):
                os.unlink(output_path)


# Required exports for ComfyUI
NODE_CLASS_MAPPINGS = {
    "MeshCraftPostProcess": MeshCraftPostProcess,
    "MeshCraftUVUnwrap": MeshCraftUVUnwrap,
    # "MeshCraftRenderMultiView": RenderMeshMultiView,  # REMOVED - use MeshCraft_RenderConditioningMaps instead (no pyglet dependency)
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MeshCraftPostProcess": "MeshCraft Post-Process",
    "MeshCraftUVUnwrap": "MeshCraft UV Unwrap",
    # "MeshCraftRenderMultiView": "Render Mesh Multi-View (Hunyuan3D)",  # REMOVED
}
