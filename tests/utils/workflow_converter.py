"""
Workflow to API JSON converter for ComfyUI.

This module converts ComfyUI UI-format workflow JSONs (exported from the UI)
into API-format JSONs (used for programmatic execution).

Format Differences:
-------------------

UI Format (exported from ComfyUI frontend):
{
  "nodes": [
    {
      "id": 14,
      "type": "Load_Trellis_Model",
      "inputs": [{"name": "...", "type": "...", "link": 12}],
      "outputs": [...],
      "widgets_values": ["image-to-3d", "flash-attn", "flash-native"],
      "pos": [x, y],
      "size": [w, h],
      ...
    }
  ],
  "links": [[link_id, from_node, from_slot, to_node, to_slot, type], ...]
}

API Format (for execution):
{
  "14": {
    "inputs": {
      "model_type": "image-to-3d",
      "attn_backend": "flash-attn",
      "spconv_algo": "flash-native",
      "some_input": ["12", 0]  // [node_id, output_slot]
    },
    "class_type": "Load_Trellis_Model",
    "_meta": {"title": "Load Trellis Model"}
  }
}
"""

import json
import copy
import urllib.request
import urllib.error
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path


# Global cache for node definitions (fetched once per session)
_NODE_DEFS_CACHE: Optional[Dict[str, Any]] = None


def fetch_node_definitions(server_address: str = "127.0.0.1:8188") -> Dict[str, Any]:
    """
    Fetch node definitions from ComfyUI's /object_info API endpoint.

    Args:
        server_address: ComfyUI server address (default: 127.0.0.1:8188)

    Returns:
        Dict mapping node class names to their definitions

    Raises:
        RuntimeError: If API is unavailable
    """
    global _NODE_DEFS_CACHE

    if _NODE_DEFS_CACHE is not None:
        return _NODE_DEFS_CACHE

    try:
        url = f"http://{server_address}/object_info"
        with urllib.request.urlopen(url, timeout=5) as response:
            _NODE_DEFS_CACHE = json.loads(response.read())
            return _NODE_DEFS_CACHE
    except (urllib.error.URLError, TimeoutError) as e:
        raise RuntimeError(
            f"Failed to fetch node definitions from {url}. "
            f"Make sure ComfyUI server is running. Error: {e}"
        )


class WorkflowConverter:
    """Converts ComfyUI workflow JSONs between UI and API formats."""

    def __init__(self, node_defs: Optional[Dict[str, Any]] = None, server_address: str = "127.0.0.1:8188"):
        """
        Initialize converter.

        Args:
            node_defs: Optional pre-fetched node definitions. If None, will fetch from API.
            server_address: ComfyUI server address for fetching node definitions
        """
        self.link_map: Dict[int, Tuple[int, int]] = {}  # link_id -> (from_node_id, from_slot)
        self.node_class_inputs: Dict[str, List[str]] = {}  # class_type -> [input_names]
        self.server_address = server_address

        # Fetch or use provided node definitions
        if node_defs is not None:
            self.node_defs = node_defs
        else:
            try:
                self.node_defs = fetch_node_definitions(server_address)
            except RuntimeError:
                # Fallback: no node definitions available
                # Will use generic parameter names
                self.node_defs = {}

    def is_api_format(self, workflow: Dict[str, Any]) -> bool:
        """
        Detect if the workflow is already in API format.

        API format has node IDs as top-level string keys.
        UI format has "nodes" and "links" arrays.
        """
        if "nodes" in workflow and "links" in workflow:
            return False

        # Check if it looks like API format (numeric string keys with class_type)
        if len(workflow) > 0:
            first_key = list(workflow.keys())[0]
            if isinstance(workflow[first_key], dict) and "class_type" in workflow[first_key]:
                return True

        return False

    def build_link_map(self, links: List[List]) -> None:
        """
        Build a mapping of link IDs to (source_node_id, source_output_slot).

        Links format: [link_id, from_node, from_slot, to_node, to_slot, type]
        """
        self.link_map = {}
        for link in links:
            link_id, from_node, from_slot, to_node, to_slot, link_type = link
            self.link_map[link_id] = (from_node, from_slot)

    def get_node_input_names(self, node: Dict[str, Any]) -> List[str]:
        """
        Extract input parameter names from a UI-format node.

        Returns list of input names in order (for matching with widgets_values).
        """
        inputs = node.get("inputs", [])

        # Inputs that are connections (have a link) don't correspond to widgets
        # Inputs that are widgets have no link or are unconnected
        widget_input_names = []

        # We need to infer widget input names from node type
        # For now, we'll use a heuristic: widgets_values correspond to
        # parameters not provided via node connections

        # This is a simplified approach - may need node-specific mappings
        return widget_input_names

    def convert_node_ui_to_api(self, node: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert a single node from UI format to API format.

        Args:
            node: UI-format node dict

        Returns:
            API-format node dict
        """
        api_node = {
            "inputs": {},
            "class_type": node["type"],
            "_meta": {
                "title": node.get("type", "Unknown")  # Could be extracted from properties
            }
        }

        # Process inputs (connections from other nodes)
        for input_def in node.get("inputs", []):
            input_name = input_def["name"]
            link_id = input_def.get("link")

            if link_id is not None and link_id in self.link_map:
                # Resolve link to [node_id, output_slot]
                from_node_id, from_slot = self.link_map[link_id]
                api_node["inputs"][input_name] = [str(from_node_id), from_slot]

        # Process widgets_values (node parameters)
        # This is tricky because we need to know parameter names
        # We'll use a node-type-specific approach
        widgets_values = node.get("widgets_values", [])

        if widgets_values:
            # Apply node-specific widget mapping
            widget_params = self._map_widgets_to_params(node["type"], widgets_values)

            # Only add widget values for parameters that are NOT already connected
            for param_name, param_value in widget_params.items():
                if param_name not in api_node["inputs"]:
                    api_node["inputs"][param_name] = param_value

        return api_node

    def _is_connection_type(self, type_spec: Any) -> bool:
        """
        Check if a parameter type is a connection (node output) rather than a widget.

        Widget types are primitive types: STRING, INT, FLOAT, BOOLEAN, COMBO
        Connection types are custom types like: IMAGE, TRIMESH, MODEL, LATENT, etc.
        """
        if isinstance(type_spec, str):
            # Widget primitive types
            WIDGET_TYPES = {"STRING", "INT", "FLOAT", "BOOLEAN", "COMBO"}
            return type_spec not in WIDGET_TYPES
        return False

    def _map_widgets_to_params(self, class_type: str, widgets_values: List[Any]) -> Dict[str, Any]:
        """
        Map widget values to parameter names using node definitions from API.

        This dynamically looks up the node's INPUT_TYPES from the ComfyUI API
        and maps widget values to parameter names in order.

        Args:
            class_type: Node class type name
            widgets_values: List of widget values from UI format

        Returns:
            Dict mapping parameter names to values
        """
        params = {}

        # Try to get node definition from API
        node_def = self.node_defs.get(class_type, {})
        if not node_def:
            # Fallback: use generic parameter names
            for i, value in enumerate(widgets_values):
                params[f"param_{i}"] = value
            return params

        # Get input definitions
        input_defs = node_def.get("input", {})
        required_inputs = input_defs.get("required", {})
        optional_inputs = input_defs.get("optional", {})

        # Use input_order if available, otherwise fall back to dict order
        input_order_def = node_def.get("input_order", {})
        required_order = input_order_def.get("required", [])
        optional_order = input_order_def.get("optional", [])

        # Combine required and optional in the correct order
        if required_order or optional_order:
            # Use explicit ordering from API
            all_param_names = required_order + optional_order
        else:
            # Fall back to dict keys (Python 3.7+ preserves insertion order)
            all_param_names = list(required_inputs.keys()) + list(optional_inputs.keys())

        # Filter to get only widget parameters (not connections)
        widget_param_names = []
        for param_name in all_param_names:
            # Get type spec from either required or optional
            type_spec = required_inputs.get(param_name) or optional_inputs.get(param_name)
            if not type_spec:
                continue

            # Check if it's a widget parameter (not a connection)
            if isinstance(type_spec, (list, tuple)) and len(type_spec) > 0:
                param_type = type_spec[0]
                if not self._is_connection_type(param_type):
                    widget_param_names.append(param_name)
            elif isinstance(type_spec, str):
                # String type specs that aren't connections
                if not self._is_connection_type(type_spec):
                    widget_param_names.append(param_name)

        # Map widget values to parameter names in order
        for i, value in enumerate(widgets_values):
            if i < len(widget_param_names):
                params[widget_param_names[i]] = value
            # Silently skip extra widget values that don't correspond to parameters
            # This can happen in UI workflows due to frontend quirks

        return params

    def convert_ui_to_api(self, workflow: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert UI-format workflow to API-format.

        Args:
            workflow: UI-format workflow dict

        Returns:
            API-format workflow dict
        """
        if self.is_api_format(workflow):
            return workflow  # Already in API format

        # Build link resolution map
        links = workflow.get("links", [])
        self.build_link_map(links)

        # Convert each node
        api_workflow = {}
        for node in workflow.get("nodes", []):
            node_id = str(node["id"])
            api_workflow[node_id] = self.convert_node_ui_to_api(node)

        return api_workflow

    def convert_file(self, input_path: Path, output_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Convert a workflow file from UI to API format.

        Args:
            input_path: Path to input workflow JSON
            output_path: Optional path to save converted workflow

        Returns:
            Converted API-format workflow dict
        """
        with open(input_path, 'r', encoding='utf-8') as f:
            workflow = json.load(f)

        api_workflow = self.convert_ui_to_api(workflow)

        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(api_workflow, f, indent=2)

        return api_workflow


def convert_workflow(workflow: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function to convert a workflow dict from UI to API format.

    Args:
        workflow: UI or API format workflow dict

    Returns:
        API-format workflow dict
    """
    converter = WorkflowConverter()
    return converter.convert_ui_to_api(workflow)


def convert_workflow_file(input_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function to convert a workflow file from UI to API format.

    Args:
        input_path: Path to input workflow JSON
        output_path: Optional path to save converted workflow

    Returns:
        API-format workflow dict
    """
    converter = WorkflowConverter()
    return converter.convert_file(Path(input_path), Path(output_path) if output_path else None)


if __name__ == "__main__":
    # Simple CLI for testing
    import sys

    if len(sys.argv) < 2:
        print("Usage: python workflow_converter.py <input_workflow.json> [output_workflow.json]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    result = convert_workflow_file(input_file, output_file)

    if not output_file:
        print(json.dumps(result, indent=2))
    else:
        print(f"Converted workflow saved to: {output_file}")
