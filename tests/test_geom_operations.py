"""
Unit tests for MeshCraft Geometric Operations node.

Tests the flexible operation sequencing and intermediate output functionality.
"""

import pytest
import trimesh
import numpy as np


class TestGeomOperationsNode:
    """Test suite for MeshCraft_GeomOperations node."""

    @pytest.fixture
    def sample_mesh(self):
        """Create a simple test mesh with some flaws for testing operations."""
        # Create a box with intentional issues
        mesh = trimesh.creation.box(extents=(1, 1, 1))

        # Add some floater geometry (separate component)
        floater = trimesh.creation.icosphere(subdivisions=1, radius=0.1)
        floater.apply_translation([2, 2, 2])

        # Combine into scene
        combined = trimesh.util.concatenate([mesh, floater])

        return combined

    @pytest.fixture
    def geom_ops_node(self):
        """Import and instantiate the GeomOperations node."""
        from nodes.geom_operations_node import MeshCraft_GeomOperations
        return MeshCraft_GeomOperations()

    def test_node_has_required_attributes(self, geom_ops_node):
        """Verify node has required ComfyUI attributes."""
        assert hasattr(geom_ops_node, 'INPUT_TYPES')
        assert hasattr(geom_ops_node, 'RETURN_TYPES')
        assert hasattr(geom_ops_node, 'RETURN_NAMES')
        assert hasattr(geom_ops_node, 'FUNCTION')
        assert hasattr(geom_ops_node, 'CATEGORY')

    def test_input_types_structure(self, geom_ops_node):
        """Verify INPUT_TYPES returns correct structure."""
        inputs = geom_ops_node.INPUT_TYPES()

        assert 'required' in inputs
        assert 'optional' in inputs

        # Check required inputs
        assert 'trimesh' in inputs['required']
        assert 'num_operations' in inputs['required']

        # Check operation slots exist
        for i in range(1, 11):
            assert f'operation_{i}' in inputs['optional']

        # Check parameter inputs
        assert 'max_facenum' in inputs['optional']
        assert 'smooth_iterations' in inputs['optional']
        assert 'lambda_factor' in inputs['optional']

    def test_return_types(self, geom_ops_node):
        """Verify node returns 10 TRIMESH outputs."""
        assert len(geom_ops_node.RETURN_TYPES) == 10
        assert all(rt == "TRIMESH" for rt in geom_ops_node.RETURN_TYPES)

        assert len(geom_ops_node.RETURN_NAMES) == 10
        for i in range(1, 11):
            assert f"mesh_after_op{i}" in geom_ops_node.RETURN_NAMES

    def test_single_operation(self, geom_ops_node, sample_mesh):
        """Test executing a single operation."""
        outputs = geom_ops_node.process(
            trimesh=sample_mesh,
            num_operations=1,
            operation_1="remove_floaters"
        )

        assert len(outputs) == 10
        assert outputs[0] is not None
        assert isinstance(outputs[0], trimesh.Trimesh)

        # After removing floaters, should have fewer vertices than original
        assert len(outputs[0].vertices) < len(sample_mesh.vertices)

    def test_multiple_operations(self, geom_ops_node, sample_mesh):
        """Test executing multiple operations in sequence."""
        outputs = geom_ops_node.process(
            trimesh=sample_mesh,
            num_operations=3,
            operation_1="remove_floaters",
            operation_2="remove_degenerate",
            operation_3="smooth_normals"
        )

        # Verify all 3 operations produced outputs
        assert outputs[0] is not None
        assert outputs[1] is not None
        assert outputs[2] is not None

        # Each should be a valid trimesh
        for i in range(3):
            assert isinstance(outputs[i], trimesh.Trimesh)
            assert len(outputs[i].vertices) > 0
            assert len(outputs[i].faces) > 0

    def test_operation_order_matters(self, geom_ops_node):
        """Verify that operation order affects the result."""
        # Create a high-poly mesh
        mesh = trimesh.creation.icosphere(subdivisions=4, radius=1.0)

        # Order 1: Reduce then smooth
        result1 = geom_ops_node.process(
            trimesh=mesh,
            num_operations=2,
            operation_1="reduce_faces",
            operation_2="smooth_normals",
            max_facenum=500
        )

        # Order 2: Smooth then reduce
        result2 = geom_ops_node.process(
            trimesh=mesh,
            num_operations=2,
            operation_1="smooth_normals",
            operation_2="reduce_faces",
            max_facenum=500
        )

        # Results should be different (different intermediate states)
        # We can't compare meshes directly, but we can check that they differ
        assert not np.array_equal(result1[1].vertices, result2[1].vertices)

    def test_skip_operation_with_none(self, geom_ops_node, sample_mesh):
        """Test that 'none' operations are skipped correctly."""
        outputs = geom_ops_node.process(
            trimesh=sample_mesh,
            num_operations=3,
            operation_1="remove_floaters",
            operation_2="none",  # Skip this slot
            operation_3="smooth_normals"
        )

        # All outputs should still be valid
        assert all(out is not None for out in outputs[:3])

        # Output 2 should be same as output 1 (since op2 was "none")
        assert len(outputs[1].vertices) == len(outputs[0].vertices)

    def test_reduce_faces_operation(self, geom_ops_node):
        """Test face reduction operation with target face count."""
        # Create high-poly mesh
        mesh = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
        initial_faces = len(mesh.faces)

        outputs = geom_ops_node.process(
            trimesh=mesh,
            num_operations=1,
            operation_1="reduce_faces",
            max_facenum=500
        )

        reduced_mesh = outputs[0]

        # Should have fewer faces than original
        assert len(reduced_mesh.faces) < initial_faces
        # Should be close to target (within reason)
        assert len(reduced_mesh.faces) <= 500 * 1.5  # Allow some margin

    def test_laplacian_smooth_operation(self, geom_ops_node, sample_mesh):
        """Test Laplacian smoothing operation."""
        outputs = geom_ops_node.process(
            trimesh=sample_mesh,
            num_operations=1,
            operation_1="laplacian_smooth",
            smooth_iterations=3,
            lambda_factor=0.5
        )

        smoothed_mesh = outputs[0]

        # Mesh should still be valid
        assert len(smoothed_mesh.vertices) > 0
        assert len(smoothed_mesh.faces) > 0

        # Vertices should have changed (smoothing modifies geometry)
        assert not np.array_equal(smoothed_mesh.vertices, sample_mesh.vertices)

    def test_all_operations_available(self, geom_ops_node, sample_mesh):
        """Test that all advertised operations work without crashing."""
        operations = [
            "remove_floaters",
            "remove_degenerate",
            "reduce_faces",
            "smooth_normals",
            "laplacian_smooth",
            "ensure_manifold",
        ]

        for op in operations:
            outputs = geom_ops_node.process(
                trimesh=sample_mesh,
                num_operations=1,
                operation_1=op,
                max_facenum=1000  # For reduce_faces
            )

            assert outputs[0] is not None, f"Operation '{op}' failed"
            assert isinstance(outputs[0], trimesh.Trimesh)

    def test_max_operations(self, geom_ops_node, sample_mesh):
        """Test using all 10 operation slots."""
        kwargs = {
            'trimesh': sample_mesh,
            'num_operations': 10,
        }

        # Set all 10 operations
        for i in range(1, 11):
            if i % 3 == 0:
                kwargs[f'operation_{i}'] = "smooth_normals"
            elif i % 3 == 1:
                kwargs[f'operation_{i}'] = "remove_degenerate"
            else:
                kwargs[f'operation_{i}'] = "none"

        outputs = geom_ops_node.process(**kwargs)

        # All 10 outputs should be populated
        assert all(out is not None for out in outputs)
        assert all(isinstance(out, trimesh.Trimesh) for out in outputs)

    def test_intermediate_outputs(self, geom_ops_node):
        """Test that intermediate outputs are correctly preserved."""
        # Start with icosphere
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)

        outputs = geom_ops_node.process(
            trimesh=mesh,
            num_operations=3,
            operation_1="reduce_faces",
            operation_2="smooth_normals",
            operation_3="laplacian_smooth",
            max_facenum=500,
            smooth_iterations=2
        )

        # Each output should have different vertex counts (approximately)
        # After reduce_faces, vertex count should be lower
        assert len(outputs[0].vertices) < len(mesh.vertices)

        # smooth_normals doesn't change geometry, only normals
        assert len(outputs[1].vertices) == len(outputs[0].vertices)

        # laplacian_smooth changes geometry
        assert not np.array_equal(outputs[2].vertices, outputs[1].vertices)

    def test_error_handling(self, geom_ops_node):
        """Test that node handles errors gracefully."""
        # Create a very simple mesh
        mesh = trimesh.creation.box(extents=(0.1, 0.1, 0.1))

        # Try to reduce to impossibly low face count
        outputs = geom_ops_node.process(
            trimesh=mesh,
            num_operations=1,
            operation_1="reduce_faces",
            max_facenum=1  # Impossibly low
        )

        # Should still return valid mesh (even if reduction didn't achieve target)
        assert outputs[0] is not None
        assert isinstance(outputs[0], trimesh.Trimesh)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
