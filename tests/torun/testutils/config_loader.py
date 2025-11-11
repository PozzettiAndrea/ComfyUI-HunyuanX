"""
Configuration loader for MeshCraft workflow tests.

Loads YAML configuration files that define which workflows to test
and which attention configurations to use.
"""

import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from .attention_configs import (
    AttentionConfig,
    get_trellis_attention_configs,
    get_hunyuan_attention_configs,
    format_config_id,
)


@dataclass
class WorkflowTestConfig:
    """Configuration for a single workflow test."""
    name: str
    enabled: bool
    timeout: int
    attention_configs: List[AttentionConfig]


class TestConfigLoader:
    """Loads and manages test configuration from YAML files."""

    DEFAULT_CONFIG_PATH = "configs/default.yaml"
    DEFAULT_TIMEOUT = 600

    def __init__(self, config_path: Optional[str] = None):
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to YAML config file (relative to tests/ directory)
                        If None, uses DEFAULT_CONFIG_PATH
        """
        if config_path is None:
            config_path = self.DEFAULT_CONFIG_PATH

        # Resolve path relative to tests directory
        tests_dir = Path(__file__).parent.parent
        self.config_file = tests_dir / config_path

        if not self.config_file.exists():
            raise FileNotFoundError(
                f"Test config file not found: {self.config_file}\n"
                f"Generate a default config with: python tests/generate_config.py"
            )

        with open(self.config_file) as f:
            self.data = yaml.safe_load(f)

        self._validate_config()

    def _validate_config(self):
        """Validate the loaded configuration structure."""
        if not isinstance(self.data, dict):
            raise ValueError(f"Config must be a dictionary, got {type(self.data)}")

        if "workflows" not in self.data:
            raise ValueError("Config must contain 'workflows' section")

        if "tests" not in self.data:
            raise ValueError("Config must contain 'tests' section")

    def get_workflow_configs(self) -> List[WorkflowTestConfig]:
        """
        Get all workflow test configurations.

        Returns:
            List of WorkflowTestConfig objects for enabled workflows
        """
        configs = []

        for workflow_name, test_spec in self.data["tests"].items():
            # Check if workflow is enabled
            workflow_settings = self.data["workflows"].get(workflow_name, {})
            if not workflow_settings.get("enabled", True):
                continue

            # Get timeout (workflow-specific or default)
            timeout = workflow_settings.get("timeout", self.DEFAULT_TIMEOUT)

            # Get model type
            model_type = workflow_settings.get("model_type")
            if not model_type:
                raise ValueError(f"Workflow {workflow_name} missing model_type")

            # Get attention configs for this workflow
            attention_configs = self._get_attention_configs(
                workflow_name,
                model_type,
                test_spec
            )

            configs.append(WorkflowTestConfig(
                name=workflow_name,
                enabled=True,
                timeout=timeout,
                attention_configs=attention_configs,
            ))

        return configs

    def _get_attention_configs(
        self,
        workflow_name: str,
        model_type: str,
        test_spec: Any
    ) -> List[AttentionConfig]:
        """
        Get attention configurations for a workflow.

        Args:
            workflow_name: Name of the workflow
            model_type: Model type (trellis or hunyuan)
            test_spec: Test specification from config (can be "all" or list of configs)

        Returns:
            List of AttentionConfig objects
        """
        # If test_spec is "all", use all available configs
        if test_spec == "all":
            if model_type == "trellis":
                return get_trellis_attention_configs()
            elif model_type == "hunyuan":
                return get_hunyuan_attention_configs()
            else:
                raise ValueError(f"Unknown model type: {model_type}")

        # Otherwise, test_spec should be a list of config names
        if not isinstance(test_spec, list):
            raise ValueError(
                f"Test spec for {workflow_name} must be 'all' or a list, got {type(test_spec)}"
            )

        # Get all available configs for this model type
        if model_type == "trellis":
            all_configs = get_trellis_attention_configs()
        elif model_type == "hunyuan":
            all_configs = get_hunyuan_attention_configs()
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Filter configs by formatted ID (e.g., "trellis-flash-attn-auto")
        selected_configs = []
        for config_name in test_spec:
            # Find matching config by format_config_id
            matching = [c for c in all_configs if format_config_id(c) == config_name]
            if not matching:
                available = [format_config_id(c) for c in all_configs]
                raise ValueError(
                    f"Unknown attention config '{config_name}' for {workflow_name}\n"
                    f"Available configs: {', '.join(available)}"
                )
            selected_configs.append(matching[0])

        return selected_configs

    def get_enabled_workflow_names(self) -> List[str]:
        """Get list of enabled workflow names."""
        return [
            name for name, settings in self.data["workflows"].items()
            if settings.get("enabled", True)
        ]


def load_test_config(config_path: Optional[str] = None) -> TestConfigLoader:
    """
    Load test configuration from YAML file.

    Args:
        config_path: Path to config file (relative to tests/ directory)
                    If None, uses default config

    Returns:
        TestConfigLoader instance
    """
    return TestConfigLoader(config_path)
