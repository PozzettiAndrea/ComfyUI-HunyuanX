"""Test utilities for ComfyUI workflow testing."""

from .workflow_converter import (
    WorkflowConverter,
    convert_workflow,
    convert_workflow_file,
)

from .attention_configs import (
    AttentionConfig,
    get_trellis_attention_configs,
    get_hunyuan_attention_configs,
    get_all_attention_configs,
    apply_attention_config,
    generate_workflow_variants,
    detect_workflow_model_type,
    get_trellis_test_params,
    get_hunyuan_test_params,
    format_config_id,
)

from .performance_tracker import (
    PerformanceMetrics,
    PerformanceTracker,
)

__all__ = [
    # Workflow converter
    "WorkflowConverter",
    "convert_workflow",
    "convert_workflow_file",
    # Attention configs
    "AttentionConfig",
    "get_trellis_attention_configs",
    "get_hunyuan_attention_configs",
    "get_all_attention_configs",
    "apply_attention_config",
    "generate_workflow_variants",
    "detect_workflow_model_type",
    "get_trellis_test_params",
    "get_hunyuan_test_params",
    "format_config_id",
    # Performance tracking
    "PerformanceMetrics",
    "PerformanceTracker",
]
