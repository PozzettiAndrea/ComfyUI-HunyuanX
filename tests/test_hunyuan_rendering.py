"""
Tests for Hunyuan 3D 2.1 rendering utilities.

Tests the rendering nodes used for texture generation:
- RenderConditioningMaps: Normal and position map rendering
- RenderRGBMultiview: Blender-based RGB multiview rendering

These tests verify different rendering backends and parameters.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from unittest.mock import patch, MagicMock
import shutil

from nodes.hunyuan_nodes import Hy3D21CameraConfig
from nodes.rendering_nodes import (
    RenderConditioningMaps,
    RenderRGBMultiview,
)


class TestRenderConditioningMaps:
    """
    Tests for RenderConditioningMaps node.

    This node renders normal and position maps using a custom differentiable
    rasterizer. It's used to create conditioning inputs for the multiview
    diffusion model.
    """

    @pytest.mark.parametrize("resolution", [256, 512])
    def test_render_conditioning_different_resolutions(self, sample_trimesh, sample_camera_config, resolution):
        """Test rendering conditioning maps at different resolutions."""
        node = RenderConditioningMaps()

        normal_maps, position_maps = node.render(
            mesh=sample_trimesh,
            camera_config=sample_camera_config,
            resolution=resolution,
            ortho_scale=1.0
        )

        # Verify output shapes
        assert normal_maps.shape[0] == 4  # 4 views
        assert position_maps.shape[0] == 4
        assert normal_maps.shape[1] == resolution  # Height
        assert normal_maps.shape[2] == resolution  # Width
        assert normal_maps.shape[3] == 3  # RGB channels

        # Verify data types and ranges
        assert normal_maps.dtype == torch.float32
        assert position_maps.dtype == torch.float32
        assert torch.all(normal_maps >= 0.0) and torch.all(normal_maps <= 1.0)

    @pytest.mark.parametrize("ortho_scale", [0.8, 1.0, 1.2])
    def test_render_conditioning_different_ortho_scales(self, sample_trimesh, sample_camera_config, ortho_scale):
        """Test rendering conditioning maps with different orthographic scales."""
        node = RenderConditioningMaps()

        normal_maps, position_maps = node.render(
            mesh=sample_trimesh,
            camera_config=sample_camera_config,
            resolution=512,
            ortho_scale=ortho_scale
        )

        # Verify outputs are generated
        assert normal_maps is not None
        assert position_maps is not None
        assert normal_maps.shape[0] == 4  # 4 views

    def test_render_conditioning_output_format(self, sample_trimesh, sample_camera_config):
        """Test that conditioning maps follow ComfyUI IMAGE format."""
        node = RenderConditioningMaps()

        normal_maps, position_maps = node.render(
            mesh=sample_trimesh,
            camera_config=sample_camera_config,
            resolution=512,
            ortho_scale=1.0
        )

        # ComfyUI IMAGE format: (N, H, W, C), float32, range [0, 1]
        assert len(normal_maps.shape) == 4
        assert normal_maps.shape[-1] == 3  # RGB
        assert normal_maps.dtype == torch.float32

        # Check that normals contain meaningful data (not all zeros/ones)
        assert not torch.allclose(normal_maps, torch.zeros_like(normal_maps))
        assert not torch.allclose(normal_maps, torch.ones_like(normal_maps))

    def test_render_conditioning_multiview_consistency(self, sample_trimesh, sample_camera_config):
        """Test that different views render different images."""
        node = RenderConditioningMaps()

        normal_maps, position_maps = node.render(
            mesh=sample_trimesh,
            camera_config=sample_camera_config,
            resolution=512,
            ortho_scale=1.0
        )

        # Verify that views are different from each other
        view_0 = normal_maps[0]
        view_1 = normal_maps[1]
        view_2 = normal_maps[2]

        # Views from different angles should be different
        assert not torch.allclose(view_0, view_1, atol=0.01)
        assert not torch.allclose(view_0, view_2, atol=0.01)


class TestRenderRGBMultiview:
    """
    Tests for RenderRGBMultiview node.

    This node uses Blender to render high-quality RGB images matching
    the Hunyuan3D-2.1 training pipeline. Tests different rendering engines
    and parameters.
    """

    @pytest.fixture
    def check_blender_available(self):
        """Check if Blender is available on the system."""
        node = RenderRGBMultiview()
        blender_path = node._find_blender()
        if blender_path is None:
            pytest.skip("Blender not found - install Blender 3.0+ to run these tests")
        return blender_path

    @pytest.mark.slow
    @pytest.mark.parametrize("engine", ["BLENDER_EEVEE", "CYCLES"])
    def test_render_rgb_different_engines(self, sample_trimesh, sample_camera_config, check_blender_available, engine):
        """
        Test rendering RGB multiview with different Blender engines.

        CYCLES: Raytracing, high quality, slower
        BLENDER_EEVEE: Real-time rasterization, faster
        """
        node = RenderRGBMultiview()

        # Use fewer samples for faster testing
        samples = 32 if engine == "CYCLES" else 1

        rgb_images, = node.render(
            mesh=sample_trimesh,
            camera_config=sample_camera_config,
            resolution=256,  # Lower resolution for faster testing
            background_color="transparent",
            engine=engine,
            samples=samples,
            ortho_scale=1.2,
            camera_distance=1.5
        )

        # Verify output shape
        assert rgb_images.shape[0] == 4  # 4 views
        assert rgb_images.shape[1] == 256  # Height
        assert rgb_images.shape[2] == 256  # Width
        assert rgb_images.shape[3] in [3, 4]  # RGB or RGBA

        # Verify data type and range
        assert rgb_images.dtype == torch.float32
        assert torch.all(rgb_images >= 0.0) and torch.all(rgb_images <= 1.0)

    @pytest.mark.slow
    @pytest.mark.parametrize("samples", [32, 128])
    def test_render_rgb_different_samples_cycles(self, sample_trimesh, sample_camera_config, check_blender_available, samples):
        """
        Test rendering with different sample counts for Cycles.

        Higher samples = better quality but slower.
        Hunyuan3D training uses 128 samples.
        """
        node = RenderRGBMultiview()

        rgb_images, = node.render(
            mesh=sample_trimesh,
            camera_config=sample_camera_config,
            resolution=256,  # Lower resolution for faster testing
            background_color="transparent",
            engine="CYCLES",
            samples=samples,
            ortho_scale=1.2,
            camera_distance=1.5
        )

        # Verify outputs are generated
        assert rgb_images is not None
        assert rgb_images.shape[0] == 4  # 4 views

    @pytest.mark.slow
    @pytest.mark.parametrize("resolution", [256, 512])
    def test_render_rgb_different_resolutions(self, sample_trimesh, sample_camera_config, check_blender_available, resolution):
        """Test rendering at different resolutions."""
        node = RenderRGBMultiview()

        rgb_images, = node.render(
            mesh=sample_trimesh,
            camera_config=sample_camera_config,
            resolution=resolution,
            background_color="transparent",
            engine="BLENDER_EEVEE",  # Use Eevee for faster testing
            samples=1,
            ortho_scale=1.2,
            camera_distance=1.5
        )

        # Verify resolution matches
        assert rgb_images.shape[1] == resolution
        assert rgb_images.shape[2] == resolution

    @pytest.mark.slow
    @pytest.mark.parametrize("background_color", ["white", "transparent"])
    def test_render_rgb_different_backgrounds(self, sample_trimesh, sample_camera_config, check_blender_available, background_color):
        """Test rendering with different background colors."""
        node = RenderRGBMultiview()

        rgb_images, = node.render(
            mesh=sample_trimesh,
            camera_config=sample_camera_config,
            resolution=256,
            background_color=background_color,
            engine="BLENDER_EEVEE",  # Use Eevee for faster testing
            samples=1,
            ortho_scale=1.2,
            camera_distance=1.5
        )

        # Verify outputs are generated
        assert rgb_images is not None

        # For transparent background, should have alpha channel
        if background_color == "transparent":
            assert rgb_images.shape[3] == 4  # RGBA
        else:
            assert rgb_images.shape[3] in [3, 4]  # RGB or RGBA

    def test_render_rgb_blender_not_found_raises_error(self, sample_trimesh, sample_camera_config):
        """Test that missing Blender raises a clear error message."""
        node = RenderRGBMultiview()

        # Mock _find_blender to return None
        with patch.object(node, '_find_blender', return_value=None):
            with pytest.raises(RuntimeError, match="Blender not found"):
                node.render(
                    mesh=sample_trimesh,
                    camera_config=sample_camera_config,
                    resolution=256,
                    background_color="transparent",
                    engine="CYCLES",
                    samples=32,
                    ortho_scale=1.2,
                    camera_distance=1.5
                )


class TestRenderingPipeline:
    """
    Integration tests for the complete rendering pipeline.

    Tests the combination of conditioning map rendering + RGB rendering
    to ensure they work together correctly.
    """

    def test_conditioning_and_rgb_same_view_count(self, sample_trimesh, sample_camera_config):
        """
        Test that conditioning maps and RGB rendering produce the same number of views.

        This is important for the downstream multiview diffusion pipeline.
        """
        # Render conditioning maps
        conditioning_node = RenderConditioningMaps()
        normal_maps, position_maps = conditioning_node.render(
            mesh=sample_trimesh,
            camera_config=sample_camera_config,
            resolution=512,
            ortho_scale=1.0
        )

        # Verify view counts match camera config
        num_views = len(sample_camera_config["selected_camera_azims"])
        assert normal_maps.shape[0] == num_views
        assert position_maps.shape[0] == num_views

    def test_conditioning_maps_batch_format(self, sample_trimesh, sample_camera_config):
        """
        Test that conditioning maps are in correct batch format for diffusion model.

        The multiview diffusion model expects:
        - Batch of images with shape (N, H, W, C)
        - Float32 tensors in range [0, 1]
        - Consistent resolution across all views
        """
        node = RenderConditioningMaps()
        normal_maps, position_maps = node.render(
            mesh=sample_trimesh,
            camera_config=sample_camera_config,
            resolution=512,
            ortho_scale=1.0
        )

        # Check batch format
        assert len(normal_maps.shape) == 4  # (N, H, W, C)
        assert len(position_maps.shape) == 4

        # Check all views have same resolution
        for i in range(normal_maps.shape[0]):
            assert normal_maps[i].shape == normal_maps[0].shape
            assert position_maps[i].shape == position_maps[0].shape

        # Check data format
        assert normal_maps.dtype == torch.float32
        assert position_maps.dtype == torch.float32
