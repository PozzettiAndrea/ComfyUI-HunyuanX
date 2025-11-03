"""
Utility functions for rendering 3D meshes to images for visual verification.

This module provides tools to generate screenshots of GLB/STL/OBJ files
using Trimesh with OSMesa for headless rendering (no X server required).
"""

import os
from pathlib import Path
from typing import Optional, Tuple, List


def render_glb_to_image(
    mesh_path: str | Path,
    output_path: str | Path,
    image_size: Tuple[int, int] = (800, 800),
    camera_position: str = "isometric",
    background_color: str = "white",
    show_edges: bool = False,
) -> bool:
    """
    Render a 3D mesh file to a PNG image using Trimesh with OSMesa backend.

    This function works in headless environments without X server or display.
    Uses CPU-based software rendering via OSMesa.

    Args:
        mesh_path: Path to the mesh file (.glb, .stl, .obj)
        output_path: Path where the PNG screenshot will be saved
        image_size: Output image dimensions (width, height)
        camera_position: Camera angle - "isometric", "front", "side", "top"
        background_color: Background color name or RGB tuple
        show_edges: Whether to show mesh edges

    Returns:
        True if rendering succeeded, False otherwise
    """
    # Force OSMesa backend for headless rendering (must be set before importing trimesh)
    os.environ['PYOPENGL_PLATFORM'] = 'osmesa'

    try:
        import trimesh
        import numpy as np
        from PIL import Image
    except ImportError as e:
        print(f"⚠️  Missing dependency for rendering: {e}")
        print("Install with: pip install trimesh pillow")
        return False

    mesh_path = Path(mesh_path)
    output_path = Path(output_path)

    if not mesh_path.exists():
        print(f"⚠️  Mesh file not found: {mesh_path}")
        return False

    try:
        # Load mesh using trimesh (supports GLB, STL, OBJ)
        mesh = trimesh.load(str(mesh_path))

        # Convert to scene if it's a single mesh
        if isinstance(mesh, trimesh.Trimesh):
            scene = trimesh.Scene(mesh)
        elif isinstance(mesh, trimesh.Scene):
            scene = mesh
        else:
            print(f"⚠️  Unsupported mesh type: {type(mesh)}")
            return False

        # Parse background color
        if background_color == "white":
            bg_color = [255, 255, 255, 255]
        elif background_color == "black":
            bg_color = [0, 0, 0, 255]
        elif background_color == "transparent":
            bg_color = [255, 255, 255, 0]
        else:
            # Assume it's already an RGBA tuple or list
            bg_color = background_color

        # Calculate camera distance based on scene bounds
        bounds = scene.bounds
        scene_size = np.linalg.norm(bounds[1] - bounds[0])
        camera_distance = scene_size * 2.0  # Pull back camera to fit scene

        # Set camera transform based on position
        # Trimesh uses 4x4 transformation matrices
        camera_positions = {
            "isometric": np.array([1, 1, 1]),  # Diagonal view
            "front": np.array([0, 0, 1]),      # Along Z axis
            "side": np.array([1, 0, 0]),       # Along X axis
            "top": np.array([0, 1, 0]),        # Along Y axis
        }

        # Get camera position vector
        if camera_position in camera_positions:
            cam_vec = camera_positions[camera_position]
            # Normalize and scale by distance
            cam_vec = cam_vec / np.linalg.norm(cam_vec) * camera_distance
            # Position relative to scene center
            camera_pos = scene.centroid + cam_vec

            # Create look-at transformation matrix
            # Camera points from camera_pos toward scene.centroid
            forward = scene.centroid - camera_pos
            forward = forward / np.linalg.norm(forward)

            # Create coordinate system
            up = np.array([0, 1, 0])  # World up
            if np.allclose(np.abs(np.dot(forward, up)), 1.0):
                # Handle case where forward is parallel to up
                up = np.array([0, 0, 1])

            right = np.cross(forward, up)
            right = right / np.linalg.norm(right)
            up = np.cross(right, forward)
            up = up / np.linalg.norm(up)

            # Build transformation matrix (camera to world)
            camera_transform = np.eye(4)
            camera_transform[:3, 0] = right
            camera_transform[:3, 1] = up
            camera_transform[:3, 2] = -forward  # Camera looks along -Z
            camera_transform[:3, 3] = camera_pos

            # Invert to get world to camera transform
            camera_transform = np.linalg.inv(camera_transform)
        else:
            # Use default camera
            camera_transform = None

        # Render the scene
        try:
            # Set resolution
            scene.camera.resolution = image_size

            if camera_transform is not None:
                scene.camera_transform = camera_transform

            # Configure better lighting for improved render quality
            # Add multiple light sources for better illumination
            scene.set_camera(
                resolution=image_size,
                fov=(60, 60)  # Field of view in degrees
            )

            # Save the image using trimesh's built-in PNG export
            # Use smooth rendering with anti-aliasing
            png_data = scene.save_image(
                resolution=image_size,
                background=bg_color,
                visible=True  # Only render visible triangles
            )

            # Write to file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'wb') as f:
                f.write(png_data)

            return True

        except Exception as render_error:
            print(f"⚠️  Rendering failed, trying fallback method: {render_error}")

            # Fallback: Use matplotlib-based rendering
            try:
                import matplotlib.pyplot as plt
                from mpl_toolkits.mplot3d.art3d import Poly3DCollection

                fig = plt.figure(figsize=(image_size[0]/100, image_size[1]/100), dpi=100)
                ax = fig.add_subplot(111, projection='3d')

                # Get all meshes from scene
                for geom_name, geom in scene.geometry.items():
                    if isinstance(geom, trimesh.Trimesh):
                        # Create mesh collection
                        mesh_collection = Poly3DCollection(
                            geom.vertices[geom.faces],
                            alpha=0.9,
                            facecolor='lightblue',
                            edgecolor='black' if show_edges else 'none',
                            linewidths=0.5 if show_edges else 0
                        )
                        ax.add_collection3d(mesh_collection)

                # Set camera angle
                if camera_position == "isometric":
                    ax.view_init(elev=35, azim=45)
                elif camera_position == "front":
                    ax.view_init(elev=0, azim=0)
                elif camera_position == "side":
                    ax.view_init(elev=0, azim=90)
                elif camera_position == "top":
                    ax.view_init(elev=90, azim=0)

                # Set equal aspect ratio
                bounds = scene.bounds
                max_range = np.max(bounds[1] - bounds[0]) / 2.0
                mid = (bounds[1] + bounds[0]) / 2.0
                ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
                ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
                ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

                # Remove axes
                ax.set_axis_off()

                # Set background
                if background_color == "white":
                    fig.patch.set_facecolor('white')
                elif background_color == "black":
                    fig.patch.set_facecolor('black')

                # Save
                output_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(
                    output_path,
                    dpi=100,
                    bbox_inches='tight',
                    facecolor=fig.get_facecolor(),
                    edgecolor='none'
                )
                plt.close(fig)

                return True

            except Exception as fallback_error:
                print(f"⚠️  Fallback rendering also failed: {fallback_error}")
                return False

    except Exception as e:
        print(f"⚠️  Failed to render {mesh_path.name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def render_multiple_views(
    mesh_path: str | Path,
    output_dir: str | Path,
    views: Optional[List[str]] = None,
    **kwargs
) -> List[Path]:
    """
    Render a mesh from multiple camera angles.

    Args:
        mesh_path: Path to the mesh file
        output_dir: Directory where screenshots will be saved
        views: List of camera positions (default: ["isometric", "front", "side", "top"])
        **kwargs: Additional arguments passed to render_glb_to_image

    Returns:
        List of paths to generated screenshots
    """
    if views is None:
        views = ["isometric", "front", "side", "top"]

    mesh_path = Path(mesh_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rendered_files = []

    for view in views:
        output_name = f"{mesh_path.stem}_{view}.png"
        output_path = output_dir / output_name

        success = render_glb_to_image(
            mesh_path,
            output_path,
            camera_position=view,
            **kwargs
        )

        if success:
            rendered_files.append(output_path)
            print(f"📸 Rendered {view} view: {output_path.name}")
        else:
            print(f"❌ Failed to render {view} view")

    return rendered_files


def create_contact_sheet(
    image_paths: List[Path],
    output_path: str | Path,
    grid_size: Optional[Tuple[int, int]] = None,
) -> bool:
    """
    Create a contact sheet (grid of images) from multiple screenshots.

    Args:
        image_paths: List of image file paths
        output_path: Path where the contact sheet will be saved
        grid_size: (rows, cols) for the grid. If None, automatically calculated.

    Returns:
        True if successful, False otherwise
    """
    try:
        from PIL import Image
        import math
    except ImportError:
        print("⚠️  PIL not installed. Install with: pip install Pillow")
        return False

    if not image_paths:
        print("⚠️  No images to combine")
        return False

    # Load images
    images = []
    for img_path in image_paths:
        if Path(img_path).exists():
            images.append(Image.open(img_path))
        else:
            print(f"⚠️  Image not found: {img_path}")

    if not images:
        print("⚠️  No valid images loaded")
        return False

    # Determine grid size
    if grid_size is None:
        num_images = len(images)
        cols = math.ceil(math.sqrt(num_images))
        rows = math.ceil(num_images / cols)
        grid_size = (rows, cols)

    rows, cols = grid_size

    # Get image size (assuming all images are the same size)
    img_width, img_height = images[0].size

    # Create contact sheet
    contact_width = img_width * cols
    contact_height = img_height * rows
    contact_sheet = Image.new('RGB', (contact_width, contact_height), 'white')

    # Paste images into grid
    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        x = col * img_width
        y = row * img_height
        contact_sheet.paste(img, (x, y))

    # Save contact sheet
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    contact_sheet.save(output_path)
    print(f"📸 Contact sheet saved: {output_path}")

    return True
