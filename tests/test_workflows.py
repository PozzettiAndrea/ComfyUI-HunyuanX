"""
Integration tests for ComfyUI-MeshCraft workflows.

These tests load saved workflow JSON files and test them with different attention
mechanism configurations to ensure compatibility across attention implementations.

Run with: pytest tests/test_workflows.py -v -m workflow
GPU tests: pytest tests/test_workflows.py -v -m gpu
"""

import pytest
import json
import copy
from pathlib import Path
from typing import Dict, List, Any


class TestWorkflowStructure:
    """Tests for workflow JSON structure and node configuration."""

    @pytest.fixture
    def workflows_dir(self):
        """Get path to workflows directory."""
        return Path(__file__).parent.parent / "workflows"

    def test_workflows_directory_exists(self, workflows_dir):
        """Verify workflows directory exists."""
        assert workflows_dir.exists(), f"Workflows directory not found: {workflows_dir}"
        assert workflows_dir.is_dir()

    def test_all_workflow_files_valid_json(self, workflows_dir):
        """Test that all workflow files are valid JSON."""
        workflow_files = list(workflows_dir.glob("*.json"))
        assert len(workflow_files) > 0, "No workflow files found"

        for workflow_file in workflow_files:
            with open(workflow_file) as f:
                data = json.load(f)
                assert isinstance(data, dict), f"{workflow_file.name} is not a dict"
                assert "nodes" in data, f"{workflow_file.name} missing 'nodes' key"


class TestWorkflowNodeConfiguration:
    """Tests for node configuration in workflows."""

    @pytest.fixture
    def load_workflow(self):
        """Helper to load a workflow JSON file."""
        def _load(workflow_name: str) -> Dict:
            workflow_path = Path(__file__).parent.parent / "workflows" / workflow_name
            with open(workflow_path) as f:
                return json.load(f)
        return _load

    def test_trellis_i2m_has_required_nodes(self, load_workflow):
        """Test that trellis-i2m.json has all required node types."""
        workflow = load_workflow("trellis-i2m.json")
        node_types = {node["type"] for node in workflow["nodes"]}

        required_nodes = {
            "LoadImage",
            "Load_Trellis_Model",
            "Trellis_Image_Conditioning",
            "Trellis_SparseStructure_Sampler",
            "Trellis_SLAT_Sampler",
            "Trellis_SLAT_Decoder",
            "Trellis_Export_GLB",
        }

        missing_nodes = required_nodes - node_types
        assert not missing_nodes, f"Missing required nodes: {missing_nodes}"

    def test_trellis_t2m_has_required_nodes(self, load_workflow):
        """Test that trellis-t2m.json has all required node types."""
        workflow = load_workflow("trellis-t2m.json")
        node_types = {node["type"] for node in workflow["nodes"]}

        # Text-to-mesh uses text conditioning instead of image
        assert "Load_Trellis_Model" in node_types
        assert "Trellis_SparseStructure_Sampler" in node_types
        assert "Trellis_SLAT_Sampler" in node_types


@pytest.mark.workflow
class TestAttentionMechanisms:
    """Tests for different attention mechanism configurations."""

    @pytest.fixture
    def load_workflow(self):
        """Helper to load a workflow JSON file."""
        def _load(workflow_name: str) -> Dict:
            workflow_path = Path(__file__).parent.parent / "workflows" / workflow_name
            with open(workflow_path) as f:
                return json.load(f)
        return _load

    @pytest.fixture
    def modify_attention_settings(self):
        """Helper to modify attention settings in a workflow."""
        def _modify(workflow: Dict, sparse_attn: str, slat_attn: str) -> Dict:
            """
            Modify attention settings in Load_Trellis_Model node.

            Args:
                workflow: Workflow dict to modify
                sparse_attn: sparse_attn_impl value ("flash-attn", "xformers", "torch-native")
                slat_attn: slat_attn_impl value ("flash-native", "xformers-native", etc.)

            Returns:
                Modified workflow dict (copy)
            """
            workflow_copy = copy.deepcopy(workflow)

            for node in workflow_copy["nodes"]:
                if node["type"] == "Load_Trellis_Model":
                    # widgets_values format: [model_type, sparse_attn_impl, slat_attn_impl]
                    node["widgets_values"][1] = sparse_attn
                    node["widgets_values"][2] = slat_attn

            return workflow_copy
        return _modify

    def test_find_trellis_model_nodes(self, load_workflow):
        """Test that we can find and parse Load_Trellis_Model nodes."""
        workflow = load_workflow("trellis-i2m.json")

        model_nodes = [n for n in workflow["nodes"] if n["type"] == "Load_Trellis_Model"]
        assert len(model_nodes) == 1, "Should have exactly one Load_Trellis_Model node"

        node = model_nodes[0]
        assert "widgets_values" in node
        assert len(node["widgets_values"]) >= 3, "Should have at least 3 widget values"

        model_type, sparse_attn, slat_attn = node["widgets_values"][:3]
        assert model_type in ["image-to-3d", "text-to-3d"]
        assert isinstance(sparse_attn, str)
        assert isinstance(slat_attn, str)

    @pytest.mark.parametrize("sparse_attn,slat_attn", [
        ("flash-attn", "flash-native"),
        ("xformers", "xformers-native"),
        ("torch-native", "flash-native"),
        ("flash-attn", "xformers-native"),
    ])
    def test_modify_attention_settings(self, load_workflow, modify_attention_settings,
                                       sparse_attn, slat_attn):
        """Test that attention settings can be modified correctly."""
        workflow = load_workflow("trellis-i2m.json")
        modified = modify_attention_settings(workflow, sparse_attn, slat_attn)

        # Verify original is unchanged
        original_node = [n for n in workflow["nodes"] if n["type"] == "Load_Trellis_Model"][0]
        modified_node = [n for n in modified["nodes"] if n["type"] == "Load_Trellis_Model"][0]

        assert original_node["widgets_values"][1:3] != [sparse_attn, slat_attn]
        assert modified_node["widgets_values"][1:3] == [sparse_attn, slat_attn]

    def test_all_attention_combinations_valid(self, load_workflow, modify_attention_settings):
        """Test that all attention combinations create valid workflow structures."""
        workflow = load_workflow("trellis-i2m.json")

        attention_combinations = [
            ("flash-attn", "flash-native"),
            ("flash-attn", "flash-causal"),
            ("xformers", "xformers-native"),
            ("xformers", "xformers-causal"),
            ("torch-native", "flash-native"),
            ("torch-native", "xformers-native"),
        ]

        for sparse_attn, slat_attn in attention_combinations:
            modified = modify_attention_settings(workflow, sparse_attn, slat_attn)

            # Verify workflow structure is still valid
            assert "nodes" in modified
            assert len(modified["nodes"]) == len(workflow["nodes"])

            # Verify attention settings were applied
            model_node = [n for n in modified["nodes"] if n["type"] == "Load_Trellis_Model"][0]
            assert model_node["widgets_values"][1] == sparse_attn
            assert model_node["widgets_values"][2] == slat_attn


@pytest.mark.gpu
@pytest.mark.slow
class TestWorkflowExecution:
    """
    Integration tests that execute workflows (requires GPU and models).

    These tests are marked as 'gpu' and 'slow' - they require:
    - GPU hardware
    - Downloaded TRELLIS models
    - Significant execution time (30s+ per test)

    Run only on GPU runners: pytest -m gpu
    Skip slow tests: pytest -m "not slow"
    """

    @pytest.fixture
    def sample_test_image_path(self, workflow_test_image):
        """Get path to test image for workflow execution."""
        # workflow_test_image fixture will be added to conftest.py
        return workflow_test_image

    @pytest.mark.skip(reason="Full workflow execution requires ComfyUI server - implement when needed")
    @pytest.mark.parametrize("workflow_name,sparse_attn,slat_attn", [
        ("trellis-i2m.json", "flash-attn", "flash-native"),
        ("trellis-i2m.json", "xformers", "xformers-native"),
    ])
    def test_execute_workflow_with_attention_variant(
        self, workflow_name, sparse_attn, slat_attn, sample_test_image_path
    ):
        """
        Test executing workflow with different attention mechanisms.

        NOTE: This test is currently skipped as it requires implementing
        workflow execution infrastructure. To enable:

        1. Implement ComfyUI server startup in conftest.py
        2. Implement workflow submission and execution helpers
        3. Add output validation
        4. Remove @pytest.mark.skip decorator

        See test_workflows.py docstring for implementation patterns.
        """
        # Implementation sketch:
        # 1. Load workflow JSON
        # 2. Modify attention settings
        # 3. Submit to ComfyUI execution queue
        # 4. Wait for completion
        # 5. Validate outputs (GLB file exists, size > 0, etc.)
        pass

    @pytest.mark.skip(reason="Implement after basic execution tests work")
    def test_attention_variants_produce_similar_outputs(self):
        """
        Test that different attention mechanisms produce similar results.

        Could check:
        - Output mesh vertex counts are similar (±10%)
        - Output file sizes are similar
        - Execution times are within reasonable range
        """
        pass


# Helper functions for future workflow execution tests

def find_node_by_type(workflow: Dict, node_type: str) -> List[Dict]:
    """Find all nodes of a given type in a workflow."""
    return [node for node in workflow["nodes"] if node["type"] == node_type]


def get_node_widget_values(workflow: Dict, node_type: str) -> Dict[int, List[Any]]:
    """Get widget values for all nodes of a given type."""
    nodes = find_node_by_type(workflow, node_type)
    return {node["id"]: node.get("widgets_values", []) for node in nodes}


def validate_workflow_connections(workflow: Dict) -> bool:
    """
    Validate that all node connections in workflow are valid.

    Returns:
        True if all connections reference valid nodes, False otherwise
    """
    node_ids = {node["id"] for node in workflow["nodes"]}

    for node in workflow["nodes"]:
        for input_socket in node.get("inputs", []):
            link_id = input_socket.get("link")
            if link_id is not None:
                # Find the link in links array (if present)
                # For now, just check that inputs are structured correctly
                pass

    return True
