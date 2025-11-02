"""
Automated testing suite for MeshCraft workflows with different attention configurations.

This test suite:
1. Loads all MeshCraft workflows (trellis-i2m, trellis-t2m, hunyuan-i2m)
2. Tests each workflow with all possible attention configurations
3. Tracks performance metrics (execution time, memory usage)
4. Validates output generation
5. Generates a test report

Total test cases: 18
- trellis-i2m: 8 attention configs
- trellis-t2m: 8 attention configs
- hunyuan-i2m: 2 attention configs
"""

from copy import deepcopy
import json
import os
import pytest
from pytest import fixture
import subprocess
import time
import torch
import uuid
import websocket
import urllib.request
import urllib.parse
from pathlib import Path
from typing import Dict, Any, Tuple

from utils import (
    convert_workflow_file,
    detect_workflow_model_type,
    apply_attention_config,
    get_trellis_attention_configs,
    get_hunyuan_attention_configs,
    format_config_id,
)


class ComfyClient:
    """Client for communicating with ComfyUI server via WebSocket and HTTP."""

    def __init__(self):
        self.client_id = None
        self.server_address = None
        self.ws = None

    def connect(
        self,
        listen: str = '127.0.0.1',
        port: int = 8188,
        client_id: str = None
    ):
        """Connect to ComfyUI server."""
        self.client_id = client_id or str(uuid.uuid4())
        self.server_address = f"{listen}:{port}"
        ws = websocket.WebSocket()
        ws.connect(f"ws://{self.server_address}/ws?clientId={self.client_id}")
        self.ws = ws

    def queue_prompt(self, prompt: Dict[str, Any]) -> Dict[str, Any]:
        """Queue a workflow for execution."""
        p = {"prompt": prompt, "client_id": self.client_id}
        data = json.dumps(p).encode('utf-8')
        req = urllib.request.Request(
            f"http://{self.server_address}/prompt",
            data=data,
            headers={'Content-Type': 'application/json'}
        )
        response = urllib.request.urlopen(req)
        return json.loads(response.read())

    def get_history(self, prompt_id: str) -> Dict[str, Any]:
        """Get execution history for a prompt."""
        url = f"http://{self.server_address}/history/{prompt_id}"
        with urllib.request.urlopen(url) as response:
            return json.loads(response.read())

    def wait_for_completion(self, prompt_id: str, timeout: int = 600) -> Dict[str, Any]:
        """
        Wait for a workflow to complete execution.

        Args:
            prompt_id: The prompt ID to wait for
            timeout: Maximum time to wait in seconds

        Returns:
            Execution history dict

        Raises:
            TimeoutError: If execution doesn't complete within timeout
        """
        start_time = time.time()

        while True:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Workflow execution timed out after {timeout}s")

            out = self.ws.recv()
            if isinstance(out, str):
                message = json.loads(out)
                if message['type'] == 'executing':
                    data = message['data']
                    if data['node'] is None and data['prompt_id'] == prompt_id:
                        # Execution is done
                        break

            time.sleep(0.1)

        return self.get_history(prompt_id)[prompt_id]

    def execute_workflow(self, workflow: Dict[str, Any], timeout: int = 600) -> Dict[str, Any]:
        """
        Execute a workflow and wait for completion.

        Args:
            workflow: API-format workflow dict
            timeout: Maximum execution time in seconds

        Returns:
            Dict with execution results and output information
        """
        # Queue the workflow
        result = self.queue_prompt(workflow)
        prompt_id = result['prompt_id']

        # Wait for completion
        history = self.wait_for_completion(prompt_id, timeout=timeout)

        # Extract output information
        outputs = history.get('outputs', {})
        output_files = []

        for node_id, node_output in outputs.items():
            # Check for various output types
            if 'images' in node_output:
                for img in node_output['images']:
                    output_files.append(img.get('filename', f"node_{node_id}_image"))

            if 'glb_path' in node_output:
                # For nodes that output GLB paths (like Trellis_Export_GLB)
                for path_data in node_output['glb_path']:
                    output_files.append(path_data)

        return {
            'prompt_id': prompt_id,
            'history': history,
            'outputs': outputs,
            'output_files': output_files,
        }


@pytest.mark.meshcraft
class TestMeshCraftWorkflows:
    """Test suite for MeshCraft workflows with attention configuration variations."""

    # Target workflows to test
    TARGET_WORKFLOWS = ["trellis-i2m", "trellis-t2m", "hunyuan-i2m"]

    @fixture(scope="class", autouse=True)
    def _server(self, args_pytest):
        """Start ComfyUI server for testing."""
        print("\n🚀 Starting ComfyUI server...")

        # Get path to ComfyUI root (3 levels up from tests directory)
        comfyui_root = Path(__file__).parent.parent.parent.parent

        p = subprocess.Popen([
            'python', str(comfyui_root / 'main.py'),
            '--output-directory', args_pytest["output_dir"],
            '--listen', args_pytest["listen"],
            '--port', str(args_pytest["port"]),
        ], cwd=str(comfyui_root))
        yield
        print("\n🛑 Stopping ComfyUI server...")
        p.kill()
        torch.cuda.empty_cache()

    def start_client(self, listen: str, port: int, retries: int = 10) -> ComfyClient:
        """
        Start and connect ComfyUI client.

        Args:
            listen: Server IP address
            port: Server port
            retries: Number of connection retry attempts

        Returns:
            Connected ComfyClient instance
        """
        client = ComfyClient()

        for i in range(retries):
            time.sleep(4)
            try:
                client.connect(listen=listen, port=port)
                print(f"✅ Connected to ComfyUI server at {listen}:{port}")
                return client
            except ConnectionRefusedError as e:
                print(f"⚠️  Connection attempt {i+1}/{retries} failed: {e}")
                if i == retries - 1:
                    raise

        return client

    @fixture(scope="class")
    def client(self, args_pytest, _server):
        """Fixture providing connected ComfyUI client."""
        client = self.start_client(args_pytest["listen"], args_pytest["port"])
        yield client
        del client

    @fixture(scope="class")
    def test_workflows(self, meshcraft_workflows, load_workflow):
        """
        Load and convert all target MeshCraft workflows.

        Returns:
            Dict mapping workflow names to (workflow_dict, model_type) tuples
        """
        workflows = {}

        for workflow_name in self.TARGET_WORKFLOWS:
            if workflow_name not in meshcraft_workflows:
                print(f"⚠️  Workflow '{workflow_name}' not found, skipping")
                continue

            workflow_path = meshcraft_workflows[workflow_name]
            workflow_dict, _ = load_workflow(workflow_path)
            model_type = detect_workflow_model_type(workflow_dict)

            workflows[workflow_name] = (workflow_dict, model_type)
            print(f"📄 Loaded workflow: {workflow_name} (model: {model_type})")

        if not workflows:
            pytest.skip("No target workflows found")

        return workflows

    def generate_test_cases(self, test_workflows):
        """
        Generate all test cases (workflow × attention config combinations).

        Returns:
            List of (workflow_name, workflow_dict, attention_config, test_id) tuples
        """
        test_cases = []

        for workflow_name, (base_workflow, model_type) in test_workflows.items():
            if model_type == "trellis":
                configs = get_trellis_attention_configs()
            elif model_type == "hunyuan":
                configs = get_hunyuan_attention_configs()
            else:
                print(f"⚠️  Unknown model type '{model_type}' for {workflow_name}")
                continue

            for config in configs:
                # Apply attention config to workflow
                try:
                    modified_workflow = apply_attention_config(base_workflow, config, inplace=False)
                    test_id = format_config_id(config)
                    test_cases.append((workflow_name, modified_workflow, config, test_id))
                except ValueError as e:
                    print(f"⚠️  Failed to apply {config} to {workflow_name}: {e}")

        return test_cases

    def test_workflow_with_attention_config(
        self,
        client,
        workflow_name,
        workflow,
        attention_config,
        test_id,
        track_workflow_performance,
        performance_tracker,
    ):
        """
        Test a workflow with a specific attention configuration.

        Args:
            client: ComfyUI client
            workflow_name: Name of the workflow
            workflow: API-format workflow dict
            attention_config: AttentionConfig object
            test_id: Test identifier string
            track_workflow_performance: Performance tracking fixture
            performance_tracker: Performance tracker instance
        """
        print(f"\n{'=' * 60}")
        print(f"Testing: {workflow_name} with {attention_config.name}")
        print(f"{'=' * 60}")

        with track_workflow_performance(
            workflow_name=workflow_name,
            attention_config=attention_config.name,
            model_type=attention_config.model_type
        ) as tracker:
            # Execute workflow
            result = client.execute_workflow(workflow, timeout=600)

            # Validate outputs
            assert 'outputs' in result, "No outputs returned from workflow"
            assert result['output_files'], "No output files generated"

            # Track output files
            tracker.add_outputs(result['output_files'])

            print(f"✅ Success! Generated {len(result['output_files'])} outputs")
            print(f"   Outputs: {result['output_files'][:3]}...")  # Show first 3

    def test_generate_report(self, performance_tracker):
        """Generate final performance report after all tests complete."""
        # This runs last due to test ordering
        print("\n" + "=" * 60)
        print("GENERATING FINAL REPORT")
        print("=" * 60)

        performance_tracker.print_summary()

        # Save results
        output_files = performance_tracker.save_results(format="both")

        print(f"\n📊 Performance reports saved:")
        for fmt, path in output_files.items():
            print(f"   {fmt.upper()}: {path}")


def pytest_generate_tests(metafunc):
    """
    Dynamically generate test parameters for parametrized tests.

    This hook is called during test collection to generate the actual test cases.
    """
    if "workflow" in metafunc.fixturenames:
        # Get the test class instance to access test_workflows
        if hasattr(metafunc, 'cls') and hasattr(metafunc.cls, 'test_workflows'):
            # This is a bit of a hack - we need access to fixtures during collection
            # For now, we'll generate test cases in a pytest hook

            # Import here to avoid circular imports
            from pathlib import Path

            # Workflow directory is relative to this test file
            workflow_dir = Path(__file__).parent.parent / "workflows"
            test_workflows_data = {}

            for workflow_name in TestMeshCraftWorkflows.TARGET_WORKFLOWS:
                workflow_path = workflow_dir / f"{workflow_name}.json"

                if not workflow_path.exists():
                    continue

                workflow_dict = convert_workflow_file(str(workflow_path))
                model_type = detect_workflow_model_type(workflow_dict)
                test_workflows_data[workflow_name] = (workflow_dict, model_type)

            # Generate test cases
            test_cases = []

            for workflow_name, (base_workflow, model_type) in test_workflows_data.items():
                if model_type == "trellis":
                    configs = get_trellis_attention_configs()
                elif model_type == "hunyuan":
                    configs = get_hunyuan_attention_configs()
                else:
                    continue

                for config in configs:
                    try:
                        modified_workflow = apply_attention_config(base_workflow, config, inplace=False)
                        test_id = format_config_id(config)
                        test_cases.append((workflow_name, modified_workflow, config, test_id))
                    except ValueError:
                        pass

            # Parametrize the test
            metafunc.parametrize(
                "workflow_name,workflow,attention_config,test_id",
                test_cases,
                ids=[f"{wf}-{tid}" for wf, _, _, tid in test_cases]
            )


if __name__ == "__main__":
    # Run tests with: pytest tests/test_workflows.py -v
    print("Run with: pytest tests/test_workflows.py -v -m meshcraft")
