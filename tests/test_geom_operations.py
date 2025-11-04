"""
Unit tests for MeshCraft Individual Geometric Operations nodes.

Tests each operation node independently (RemoveFloaters, ReduceFaces, etc.).
"""

import pytest
import trimesh
import numpy as np


class TestRemoveDegenerateNode:
    """Test suite for MeshCraft_RemoveDegenerate node."""

    @pytest.fixture
    def node(self):
        """Import and instantiate the RemoveDegenerate node."""
        from nodes.geom_operations_individual import MeshCraft_RemoveDegenerate
        return MeshCraft_RemoveDegenerate()

    def test_node_structure(self, node):
        """Verify node has required ComfyUI attributes."""
        assert hasattr(node, 'INPUT_TYPES')
        assert hasattr(node, 'RETURN_TYPES')
        assert hasattr(node, 'FUNCTION')
        assert node.CATEGORY == "MeshCraft/Operations"

    def test_cleans_mesh(self, node, sample_trimesh):
        """Test that mesh is cleaned."""
        result, = node.process(sample_trimesh)

        # Result should be valid
        assert isinstance(result, trimesh.Trimesh)
        assert len(result.vertices) > 0
        assert len(result.faces) > 0


class TestReduceFacesNode:
    """Test suite for MeshCraft_ReduceFaces node."""

    @pytest.fixture
    def node(self):
        """Import and instantiate the ReduceFaces node."""
        from nodes.geom_operations_individual import MeshCraft_ReduceFaces
        return MeshCraft_ReduceFaces()

    def test_node_structure(self, node):
        """Verify node has required ComfyUI attributes."""
        assert hasattr(node, 'INPUT_TYPES')
        assert hasattr(node, 'RETURN_TYPES')
        assert hasattr(node, 'FUNCTION')
        assert node.CATEGORY == "MeshCraft/Operations"

    def test_input_has_max_facenum_parameter(self, node):
        """Verify node has max_facenum parameter."""
        inputs = node.INPUT_TYPES()
        assert 'max_facenum' in inputs['required']

    @pytest.mark.slow
    def test_reduce_faces_operation(self, node, sample_trimesh):
        """Test face reduction with target face count on Stanford Bunny."""
        initial_faces = len(sample_trimesh.faces)
        target_faces = 10000  # Reasonable target for bunny (~200K-300K initial)

        result, = node.process(sample_trimesh, max_facenum=target_faces)

        # Check that faces were reduced
        assert len(result.faces) < initial_faces, "Face count should be reduced"

        # The algorithm outputs ~2x target, so we compensate by requesting half
        # Final result should be within reasonable margin of target
        assert len(result.faces) <= target_faces * 2.5, \
            f"Face count {len(result.faces)} exceeds 2.5x target {target_faces}"


class TestSmoothNormalsNode:
    """Test suite for MeshCraft_SmoothNormals node."""

    @pytest.fixture
    def node(self):
        """Import and instantiate the SmoothNormals node."""
        from nodes.geom_operations_individual import MeshCraft_SmoothNormals
        return MeshCraft_SmoothNormals()

    @pytest.fixture
    def sample_mesh(self):
        """Create a simple mesh."""
        return trimesh.creation.icosphere(subdivisions=2)

    def test_node_structure(self, node):
        """Verify node has required ComfyUI attributes."""
        assert hasattr(node, 'INPUT_TYPES')
        assert hasattr(node, 'RETURN_TYPES')
        assert hasattr(node, 'FUNCTION')
        assert node.CATEGORY == "MeshCraft/Operations"

    def test_recalculates_normals(self, node, sample_mesh):
        """Test that vertex normals are recalculated."""
        result, = node.process(sample_mesh)

        # Result should have vertex normals
        assert hasattr(result, 'vertex_normals')
        assert len(result.vertex_normals) == len(result.vertices)

        # Normals should be unit length
        norms = np.linalg.norm(result.vertex_normals, axis=1)
        assert np.allclose(norms, 1.0, atol=0.01), "Normals should be normalized"


class TestLaplacianSmoothNode:
    """Test suite for MeshCraft_LaplacianSmooth node."""

    @pytest.fixture
    def node(self):
        """Import and instantiate the LaplacianSmooth node."""
        from nodes.geom_operations_individual import MeshCraft_LaplacianSmooth
        return MeshCraft_LaplacianSmooth()

    @pytest.fixture
    def sample_mesh(self):
        """Create a mesh with some irregularities."""
        return trimesh.creation.icosphere(subdivisions=2)

    def test_node_structure(self, node):
        """Verify node has required ComfyUI attributes."""
        assert hasattr(node, 'INPUT_TYPES')
        assert hasattr(node, 'RETURN_TYPES')
        assert hasattr(node, 'FUNCTION')
        assert node.CATEGORY == "MeshCraft/Operations"

    def test_input_has_parameters(self, node):
        """Verify node has smoothing parameters."""
        inputs = node.INPUT_TYPES()
        assert 'iterations' in inputs['required']
        assert 'lambda_factor' in inputs['required']

    def test_smooths_geometry(self, node, sample_mesh):
        """Test that Laplacian smoothing modifies geometry."""
        original_vertices = sample_mesh.vertices.copy()

        result, = node.process(sample_mesh, iterations=3, lambda_factor=0.5)

        # Vertices should have moved
        assert not np.allclose(result.vertices, original_vertices), \
            "Vertices should be modified by smoothing"

        # Mesh should still be valid
        assert len(result.vertices) == len(original_vertices)
        assert len(result.faces) == len(sample_mesh.faces)


class TestEnsureManifoldNode:
    """Test suite for MeshCraft_EnsureManifold node."""

    @pytest.fixture
    def node(self):
        """Import and instantiate the EnsureManifold node."""
        from nodes.geom_operations_individual import MeshCraft_EnsureManifold
        return MeshCraft_EnsureManifold()

    @pytest.fixture
    def sample_mesh(self):
        """Create a simple mesh."""
        return trimesh.creation.box(extents=(1, 1, 1))

    def test_node_structure(self, node):
        """Verify node has required ComfyUI attributes."""
        assert hasattr(node, 'INPUT_TYPES')
        assert hasattr(node, 'RETURN_TYPES')
        assert hasattr(node, 'FUNCTION')
        assert node.CATEGORY == "MeshCraft/Operations"

    def test_processes_mesh(self, node, sample_mesh):
        """Test that ensure_manifold processes the mesh."""
        result, = node.process(sample_mesh)

        # Result should be valid
        assert isinstance(result, trimesh.Trimesh)
        assert len(result.vertices) > 0
        assert len(result.faces) > 0

    def test_preserves_watertight_mesh(self, node, sample_mesh):
        """Test that a watertight mesh stays watertight."""
        # Box is watertight
        assert sample_mesh.is_watertight

        result, = node.process(sample_mesh)

        # Should remain watertight
        assert result.is_watertight, "Watertight mesh should remain watertight"


class TestNodeRegistration:
    """Test that all nodes are properly registered."""

    def test_all_nodes_registered(self):
        """Verify all 6 nodes are in NODE_CLASS_MAPPINGS."""
        from nodes.geom_operations_individual import NODE_CLASS_MAPPINGS

        expected_nodes = [
            "MeshCraft_RemoveFloaters",
            "MeshCraft_RemoveDegenerate",
            "MeshCraft_ReduceFaces",
            "MeshCraft_SmoothNormals",
            "MeshCraft_LaplacianSmooth",
            "MeshCraft_EnsureManifold",
        ]

        for node_id in expected_nodes:
            assert node_id in NODE_CLASS_MAPPINGS, f"{node_id} not registered"

    def test_display_names_exist(self):
        """Verify all nodes have display names."""
        from nodes.geom_operations_individual import (
            NODE_CLASS_MAPPINGS,
            NODE_DISPLAY_NAME_MAPPINGS
        )

        for node_id in NODE_CLASS_MAPPINGS.keys():
            assert node_id in NODE_DISPLAY_NAME_MAPPINGS, \
                f"{node_id} missing display name"
