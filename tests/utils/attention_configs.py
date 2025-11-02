"""
Attention configuration generator for ComfyUI MeshCraft workflow testing.

This module defines attention configurations for different model types and provides
utilities to modify workflows with specific attention settings.
"""

from typing import Dict, List, Tuple, Any
from copy import deepcopy


# Trellis Model Attention Configurations
TRELLIS_ATTN_BACKENDS = ["flash-attn", "sdpa", "naive", "xformers"]
TRELLIS_SPCONV_ALGOS = ["auto", "flash-native"]

# Hunyuan Model Attention Configurations
HUNYUAN_ATTENTION_MODES = ["sdpa", "sageattn"]


class AttentionConfig:
    """Represents an attention configuration for a specific model type."""

    def __init__(self, model_type: str, config: Dict[str, Any], name: str = None):
        """
        Initialize attention configuration.

        Args:
            model_type: Type of model ("trellis" or "hunyuan")
            config: Configuration dict with attention parameters
            name: Optional human-readable name for the config
        """
        self.model_type = model_type
        self.config = config
        self.name = name or self._generate_name()

    def _generate_name(self) -> str:
        """Generate a name from the configuration."""
        if self.model_type == "trellis":
            return f"{self.config['attn_backend']}_{self.config['spconv_algo']}"
        elif self.model_type == "hunyuan":
            return f"{self.config['attention_mode']}"
        return "unknown"

    def __str__(self) -> str:
        return f"{self.model_type}:{self.name}"

    def __repr__(self) -> str:
        return f"AttentionConfig(model_type='{self.model_type}', config={self.config}, name='{self.name}')"


def get_trellis_attention_configs() -> List[AttentionConfig]:
    """
    Get all possible Trellis attention configurations.

    Returns:
        List of AttentionConfig objects for Trellis models
    """
    configs = []
    for attn_backend in TRELLIS_ATTN_BACKENDS:
        for spconv_algo in TRELLIS_SPCONV_ALGOS:
            config = {
                "attn_backend": attn_backend,
                "spconv_algo": spconv_algo
            }
            configs.append(AttentionConfig("trellis", config))
    return configs


def get_hunyuan_attention_configs() -> List[AttentionConfig]:
    """
    Get all possible Hunyuan attention configurations.

    Returns:
        List of AttentionConfig objects for Hunyuan models
    """
    configs = []
    for attention_mode in HUNYUAN_ATTENTION_MODES:
        config = {"attention_mode": attention_mode}
        configs.append(AttentionConfig("hunyuan", config))
    return configs


def get_all_attention_configs() -> Dict[str, List[AttentionConfig]]:
    """
    Get all attention configurations for all model types.

    Returns:
        Dict mapping model types to lists of AttentionConfig objects
    """
    return {
        "trellis": get_trellis_attention_configs(),
        "hunyuan": get_hunyuan_attention_configs()
    }


def find_node_by_class_type(workflow: Dict[str, Any], class_type: str) -> Tuple[str, Dict[str, Any]]:
    """
    Find a node in the workflow by its class_type.

    Args:
        workflow: API-format workflow dict
        class_type: Node class type to find (e.g., "Load_Trellis_Model")

    Returns:
        Tuple of (node_id, node_dict) or (None, None) if not found
    """
    for node_id, node in workflow.items():
        if node.get("class_type") == class_type:
            return node_id, node
    return None, None


def apply_trellis_attention_config(
    workflow: Dict[str, Any],
    attn_backend: str,
    spconv_algo: str,
    inplace: bool = False
) -> Dict[str, Any]:
    """
    Apply Trellis attention configuration to a workflow.

    Modifies the Load_Trellis_Model node with the specified attention settings.

    Args:
        workflow: API-format workflow dict
        attn_backend: Attention backend ("flash-attn", "sdpa", "naive", "xformers")
        spconv_algo: Sparse convolution algorithm ("auto", "flash-native")
        inplace: If True, modify workflow in place; otherwise, create a copy

    Returns:
        Modified workflow dict

    Raises:
        ValueError: If Load_Trellis_Model node is not found in workflow
    """
    wf = workflow if inplace else deepcopy(workflow)

    node_id, node = find_node_by_class_type(wf, "Load_Trellis_Model")

    if node_id is None:
        raise ValueError("Load_Trellis_Model node not found in workflow")

    # Update attention configuration
    node["inputs"]["attn_backend"] = attn_backend
    node["inputs"]["spconv_algo"] = spconv_algo

    return wf


def apply_hunyuan_attention_config(
    workflow: Dict[str, Any],
    attention_mode: str,
    inplace: bool = False
) -> Dict[str, Any]:
    """
    Apply Hunyuan attention configuration to a workflow.

    Modifies the LoadHunyuanDiT node with the specified attention setting.

    Args:
        workflow: API-format workflow dict
        attention_mode: Attention mode ("sdpa", "sageattn")
        inplace: If True, modify workflow in place; otherwise, create a copy

    Returns:
        Modified workflow dict

    Raises:
        ValueError: If LoadHunyuanDiT node is not found in workflow
    """
    wf = workflow if inplace else deepcopy(workflow)

    node_id, node = find_node_by_class_type(wf, "LoadHunyuanDiT")

    if node_id is None:
        raise ValueError("LoadHunyuanDiT node not found in workflow")

    # Update attention configuration
    node["inputs"]["attention_mode"] = attention_mode

    return wf


def apply_attention_config(
    workflow: Dict[str, Any],
    config: AttentionConfig,
    inplace: bool = False
) -> Dict[str, Any]:
    """
    Apply an AttentionConfig to a workflow.

    Args:
        workflow: API-format workflow dict
        config: AttentionConfig object to apply
        inplace: If True, modify workflow in place; otherwise, create a copy

    Returns:
        Modified workflow dict

    Raises:
        ValueError: If model type is unknown or required node is not found
    """
    if config.model_type == "trellis":
        return apply_trellis_attention_config(
            workflow,
            config.config["attn_backend"],
            config.config["spconv_algo"],
            inplace=inplace
        )
    elif config.model_type == "hunyuan":
        return apply_hunyuan_attention_config(
            workflow,
            config.config["attention_mode"],
            inplace=inplace
        )
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")


def generate_workflow_variants(
    workflow: Dict[str, Any],
    model_type: str
) -> List[Tuple[AttentionConfig, Dict[str, Any]]]:
    """
    Generate all workflow variants with different attention configurations.

    Args:
        workflow: Base workflow dict (API format)
        model_type: Model type ("trellis" or "hunyuan")

    Returns:
        List of tuples: (AttentionConfig, modified_workflow_dict)
    """
    variants = []

    if model_type == "trellis":
        configs = get_trellis_attention_configs()
    elif model_type == "hunyuan":
        configs = get_hunyuan_attention_configs()
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    for config in configs:
        try:
            modified_workflow = apply_attention_config(workflow, config, inplace=False)
            variants.append((config, modified_workflow))
        except ValueError as e:
            # Skip if the workflow doesn't contain the required nodes
            print(f"Warning: Skipping {config} - {e}")
            continue

    return variants


def detect_workflow_model_type(workflow: Dict[str, Any]) -> str:
    """
    Detect the model type used in a workflow.

    Args:
        workflow: API-format workflow dict

    Returns:
        "trellis", "hunyuan", or "unknown"
    """
    for node_id, node in workflow.items():
        class_type = node.get("class_type", "")

        if "Trellis" in class_type:
            return "trellis"
        elif "Hunyuan" in class_type or "Hy3D" in class_type:
            return "hunyuan"

    return "unknown"


# Test parameter generation for pytest.mark.parametrize
def get_trellis_test_params() -> List[Tuple[str, str]]:
    """
    Get test parameters for Trellis attention configurations.

    Returns:
        List of (attn_backend, spconv_algo) tuples
    """
    params = []
    for attn_backend in TRELLIS_ATTN_BACKENDS:
        for spconv_algo in TRELLIS_SPCONV_ALGOS:
            params.append((attn_backend, spconv_algo))
    return params


def get_hunyuan_test_params() -> List[str]:
    """
    Get test parameters for Hunyuan attention configurations.

    Returns:
        List of attention_mode strings
    """
    return HUNYUAN_ATTENTION_MODES.copy()


def format_config_id(config: AttentionConfig) -> str:
    """
    Format an AttentionConfig as a pytest test ID.

    Args:
        config: AttentionConfig object

    Returns:
        String formatted for pytest test IDs (e.g., "trellis-flash_attn-flash_native")
    """
    if config.model_type == "trellis":
        return f"trellis-{config.config['attn_backend']}-{config.config['spconv_algo']}"
    elif config.model_type == "hunyuan":
        return f"hunyuan-{config.config['attention_mode']}"
    return "unknown"


if __name__ == "__main__":
    # Print all configurations
    print("=== Trellis Attention Configurations ===")
    for config in get_trellis_attention_configs():
        print(f"  {config.name}: {config.config}")

    print(f"\nTotal Trellis configs: {len(get_trellis_attention_configs())}")

    print("\n=== Hunyuan Attention Configurations ===")
    for config in get_hunyuan_attention_configs():
        print(f"  {config.name}: {config.config}")

    print(f"\nTotal Hunyuan configs: {len(get_hunyuan_attention_configs())}")

    print("\n=== Pytest Test Parameters ===")
    print("Trellis:", get_trellis_test_params())
    print("Hunyuan:", get_hunyuan_test_params())
