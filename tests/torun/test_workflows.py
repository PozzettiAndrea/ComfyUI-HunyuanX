"""
Automated testing suite for HunyuanX workflows with different attention configurations.

This test suite:
1. Loads HunyuanX workflows (hunyuan-i2m)
2. Tests each workflow with all possible attention configurations
3. Tracks performance metrics (execution time, memory usage)
4. Validates output generation
5. Generates a test report
6. Saves outputs to organized folders with automatic screenshots

Total test cases: 2
- hunyuan-i2m: 2 attention configs (sdpa, flash_attn)

Output Structure:
    tests/test_outputs/
    ├── latest → 2025-11-03_14-30-00 (symlink to most recent run)
    ├── 2025-11-03_14-30-00/
    │   └── hunyuan-i2m/
    │       ├── sdpa.stl
    │       ├── sdpa_preview.png
    │       ├── flash_attn.stl
    │       └── flash_attn_preview.png
    └── 2025-11-03_13-15-00/ (previous run, preserved)
        └── ...

Each test run creates:
- Timestamped folder to preserve test history
- Workflow-specific subdirectories
- 3D output files (.stl, .obj, .glb)
- PNG screenshot previews for visual verification
- "latest" symlink for easy access to most recent results
"""

import json
import os
import pytest
from pytest import fixture
import shutil
import time
import uuid
import websocket
import urllib.request
import urllib.parse
from pathlib import Path
from typing import Dict, Any

from testutils import (
    convert_workflow_file,
    detect_workflow_model_type,
    apply_attention_config,
    format_config_id,
)
from testutils.config_loader import load_test_config
from render_utils import render_glb_to_image

import threading
workflow_lock = threading.Lock()

class ComfyClient:
    """Client for communicating with ComfyUI server via WebSocket and HTTP."""

    def __init__(self):
        self.client_id = None
        self.server_address = None
        self.ws = None

    def connect(
        self,
        listen: str = '0.0.0.0',
        port: int = 8188,
        client_id: str = None
    ):
        """Connect to ComfyUI server."""
        self.client_id = client_id or str(uuid.uuid4())
        self.server_address = f"{listen}:{port}"
        ws = websocket.WebSocket()
        ws.connect(f"ws://{self.server_address}/ws?clientId={self.client_id}")
        self.ws = ws

    def wait_for_registry_ready(self, timeout=300):
        """Wait until ComfyUI-Manager registry is loaded."""
        start = time.time()
        while time.time() - start < timeout:
            try:
                with urllib.request.urlopen(f"http://{self.server_address}/extensions") as resp:
                    data = json.loads(resp.read())
                    # 'ComfyUI-Manager' should appear once the registry is ready
                    if any("ComfyUI-Manager" in ext for ext in data):
                        print("🟢 ComfyUI registry ready.")
                        return
            except Exception:
                pass
            print("⏳ Waiting for ComfyUI registry to load...")
            time.sleep(5)
        raise TimeoutError("ComfyUI registry never became ready.")

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

    def wait_for_idle(self, poll_interval: float = 5.0, timeout: int = 600):
        """
        Wait until ComfyUI's queue is empty (no active jobs).
        Uses /queue endpoint to check status.
        """
        start = time.time()
        while True:
            try:
                with urllib.request.urlopen(f"http://{self.server_address}/queue") as resp:
                    data = json.loads(resp.read())
                    if not data.get("queue_running") and not data.get("queue_pending"):
                        print("🟢 ComfyUI queue is idle.")
                        break
            except Exception:
                pass
    
            if time.time() - start > timeout:
                raise TimeoutError("Timed out waiting for ComfyUI to become idle.")
            time.sleep(poll_interval)

    def execute_workflow(self, workflow: Dict[str, Any], timeout: int = 600,
                        workflow_name: str = None, attention_config_name: str = None,
                        test_run_id: str = None) -> Dict[str, Any]:
        """
        Execute a workflow and wait for completion.

        Args:
            workflow: API-format workflow dict
            timeout: Maximum execution time in seconds
            workflow_name: Name of the workflow being tested (for output naming)
            attention_config_name: Attention config being tested (for output naming)
            test_run_id: Unique ID for this test run (for organizing outputs)

        Returns:
            Dict with execution results and output information
        """
        # Queue the workflow
        result = self.queue_prompt(workflow)
        prompt_id = result['prompt_id']

        # Wait for completion
        history = self.wait_for_completion(prompt_id, timeout=timeout)

        # 🧠 Wait for ComfyUI to fully finish before starting the next workflow
        self.wait_for_idle(poll_interval=10, timeout=3600)

        # Extract output information
        outputs = history.get('outputs', {})
        output_files = []

        # Debug output (only if no outputs found)
        if not outputs:
            import json
            print(f"\n{'='*60}")
            print(f"⚠️  WARNING: No outputs in history!")
            print(f"History keys: {list(history.keys())}")
            print(f"{'='*60}\n")

        for node_id, node_output in outputs.items():
            # Check for various output types
            if 'images' in node_output:
                for img in node_output['images']:
                    output_files.append(img.get('filename', f"node_{node_id}_image"))

            # Special handling for Preview3D nodes which use ui.result format
            if 'ui' in node_output:
                ui_data = node_output['ui']
                if isinstance(ui_data, dict) and 'result' in ui_data:
                    result = ui_data['result']
                    if isinstance(result, (list, tuple)):
                        for v in result:
                            if v is not None and isinstance(v, str) and len(v) > 0:
                                output_files.append(str(v))
                    elif isinstance(result, str) and len(result) > 0:
                        output_files.append(result)

            # Look for any output that contains file paths
            # Common patterns: result, glb_path, ply_path, mesh_path, file_path, etc.
            for key, value in node_output.items():
                # Skip already processed outputs
                if key in ['images', 'ui']:
                    continue

                # Check if this looks like a file path output
                if 'path' in key.lower() or 'file' in key.lower() or key == 'result':
                    if isinstance(value, (list, tuple)):
                        # Extract non-null values that look like file paths
                        for v in value:
                            if v is not None and isinstance(v, str) and len(v) > 0:
                                output_files.append(str(v))
                    elif isinstance(value, str) and len(value) > 0:
                        output_files.append(value)

        # Debug: Print details if no files were extracted
        if outputs and not output_files:
            import json
            print(f"\n⚠️  WARNING: Outputs exist but no files extracted!")
            print(f"Output structure: {json.dumps(outputs, indent=2, default=str)[:1000]}\n")

        # Copy outputs to test directory with descriptive names
        if output_files and workflow_name and attention_config_name:
            # Create granular folder structure: output/{run_id}/{workflow_name}/
            base_output_dir = Path(__file__).parent / "output"
            if test_run_id:
                test_output_dir = base_output_dir / test_run_id / workflow_name
            else:
                # Fallback if no run ID provided
                test_output_dir = base_output_dir / workflow_name
            test_output_dir.mkdir(parents=True, exist_ok=True)

            # Get ComfyUI output directory for resolving relative paths
            try:
                # Try to import ComfyUI's folder_paths module
                import sys
                comfy_path = Path(__file__).parent.parent.parent.parent
                if str(comfy_path) not in sys.path:
                    sys.path.insert(0, str(comfy_path))
                import folder_paths
                comfy_output_dir = Path(folder_paths.get_output_directory())
            except (ImportError, Exception):
                # Fallback: assume default ComfyUI output structure
                comfy_output_dir = Path(__file__).parent.parent.parent.parent / "output"
                if not comfy_output_dir.exists():
                    print(f"⚠️  Warning: Could not determine ComfyUI output directory")
                    comfy_output_dir = Path.cwd()

            # Supported 3D file formats
            supported_formats = ('.glb', '.stl', '.obj', '.ply', '.fbx')

            for output_file in output_files:
                if not isinstance(output_file, str):
                    continue

                # Check if this is a 3D file
                if not output_file.lower().endswith(supported_formats):
                    continue

                # Handle both absolute and relative paths
                src = Path(output_file)
                if not src.is_absolute():
                    # Try to resolve relative path from ComfyUI output directory
                    src = comfy_output_dir / output_file
                    if not src.exists():
                        # Also try current working directory
                        src = Path(output_file)

                if src.exists():
                    # Create descriptive filename: config_name.ext
                    file_ext = src.suffix
                    test_name = f"{attention_config_name}{file_ext}"
                    dst = test_output_dir / test_name
                    shutil.copy2(src, dst)

                    # Print relative path from output directory
                    rel_path = dst.relative_to(base_output_dir)
                    print(f"📁 Copied output: {rel_path}")

                    # Generate screenshot for visual verification (skip if in headless env)
                    if file_ext.lower() in ('.glb', '.stl', '.obj'):
                        screenshot_name = f"{attention_config_name}_preview.png"
                        screenshot_path = test_output_dir / screenshot_name
                        try:
                            import signal

                            # Set timeout for rendering (max 10 seconds)
                            def timeout_handler(signum, frame):
                                raise TimeoutError("Screenshot rendering timed out")

                            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                            signal.alarm(10)  # 10 second timeout

                            try:
                                success = render_glb_to_image(
                                    dst,
                                    screenshot_path,
                                    camera_position="isometric",
                                    background_color="white",
                                    show_edges=False
                                )
                                if success:
                                    rel_screenshot = screenshot_path.relative_to(base_output_dir)
                                    print(f"📸 Generated screenshot: {rel_screenshot}")
                            finally:
                                signal.alarm(0)  # Cancel alarm
                                signal.signal(signal.SIGALRM, old_handler)

                        except (Exception, TimeoutError) as e:
                            print(f"⚠️  Screenshot skipped (headless env?): {type(e).__name__}")
                else:
                    print(f"⚠️  Output file not found: {output_file}")

        return {
            'prompt_id': prompt_id,
            'history': history,
            'outputs': outputs,
            'output_files': output_files,
        }


@pytest.mark.meshcraft
class TestMeshCraftWorkflows:
    """
    Test suite for MeshCraft workflows with attention configuration variations.

    Test cases are generated from YAML configuration files in tests/configs/.
    """

    @staticmethod
    def check_server_running(listen: str = '0.0.0.0', port: int = 8188) -> bool:
        """
        Check if ComfyUI server is running.

        Args:
            listen: Server IP address
            port: Server port

        Returns:
            True if server is accessible, False otherwise
        """
        try:
            url = f"http://{listen}:{port}/system_stats"
            urllib.request.urlopen(url, timeout=2)
            return True
        except (urllib.error.URLError, TimeoutError):
            return False

    def start_client(self, listen: str, port: int, retries: int = 3) -> ComfyClient:
        """
        Connect to an already-running ComfyUI server.

        Args:
            listen: Server IP address
            port: Server port
            retries: Number of connection retry attempts

        Returns:
            Connected ComfyClient instance

        Raises:
            RuntimeError: If ComfyUI server is not running
        """
        # First check if server is running at all
        if not self.check_server_running(listen, port):
            raise RuntimeError(
                f"\n\n❌ ComfyUI server is not running at {listen}:{port}\n\n"
                "Please start the ComfyUI server manually before running tests:\n"
                f"  python main.py --listen {listen} --port {port}\n\n"
                "Then run the tests in a separate terminal:\n"
                "  ./run_tests.sh test_workflows.py -v -m meshcraft\n"
            )

        # Server is running, now connect WebSocket client
        client = ComfyClient()

        for i in range(retries):
            if i > 0:
                time.sleep(2)
            try:
                client.connect(listen=listen, port=port)
                client.wait_for_registry_ready(timeout=300)
                print(f"✅ Connected to ComfyUI server at {listen}:{port}")
                return client
            except (ConnectionRefusedError, OSError) as e:
                print(f"⚠️  Connection attempt {i+1}/{retries} failed: {e}")
                if i == retries - 1:
                    raise RuntimeError(
                        f"Failed to connect WebSocket to {listen}:{port} after {retries} attempts"
                    ) from e

        return client

    @fixture(scope="class")
    def client(self, args_pytest):
        """Fixture providing connected ComfyUI client (requires server already running)."""
        client = self.start_client(args_pytest["listen"], args_pytest["port"])
        yield client
        del client


    @pytest.mark.order("sequential")
    def test_workflow_with_attention_config(
        self,
        client,
        workflow_name,
        workflow,
        attention_config,
        test_id,
        timeout,
        track_workflow_performance,
        performance_tracker,
        test_run_id,
    ):
        print(f"\n{'=' * 60}")
        print(f"Testing: {workflow_name} with {attention_config.name}")
        print(f"Timeout: {timeout}s")
        print(f"{'=' * 60}")

        # 🚦 Absolute sequential guard
        with workflow_lock:
            with track_workflow_performance(
                workflow_name=workflow_name,
                attention_config=attention_config.name,
                model_type=attention_config.model_type
            ) as tracker:
                result = client.execute_workflow(
                    workflow,
                    timeout=timeout,
                    workflow_name=workflow_name,
                    attention_config_name=attention_config.name,
                    test_run_id=test_run_id
                )

                assert 'outputs' in result, "No outputs returned from workflow"
                assert result['output_files'], "No output files generated"
                tracker.add_outputs(result['output_files'])


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
    Uses the YAML configuration file specified via --config option.
    """
    if "workflow" in metafunc.fixturenames:
        # Check if this is the workflow test function
        if hasattr(metafunc, 'cls') and metafunc.cls == TestMeshCraftWorkflows:
            # Load test configuration
            config_path = metafunc.config.getoption("--config")

            try:
                config_loader = load_test_config(config_path)
            except FileNotFoundError as e:
                # Config file doesn't exist - provide helpful error
                pytest.fail(
                    f"{e}\n\n"
                    f"Generate a config file with:\n"
                    f"  cd tests && python generate_config.py\n"
                )
                return

            # Workflow directory is relative to this test file
            workflow_dir = Path(__file__).parent.parent / "workflows"

            # Generate test cases from config
            test_cases = []

            for workflow_config in config_loader.get_workflow_configs():
                workflow_name = workflow_config.name
                workflow_path = workflow_dir / f"{workflow_name}.json"

                if not workflow_path.exists():
                    print(f"⚠️  Workflow file not found: {workflow_path}")
                    continue

                # Load and convert workflow
                workflow_dict = convert_workflow_file(str(workflow_path))

                # Generate test cases for each attention config
                for attention_config in workflow_config.attention_configs:
                    try:
                        modified_workflow = apply_attention_config(
                            workflow_dict,
                            attention_config,
                            inplace=False
                        )
                        test_id = format_config_id(attention_config)
                        test_cases.append((
                            workflow_name,
                            modified_workflow,
                            attention_config,
                            test_id,
                            workflow_config.timeout,
                        ))
                    except ValueError as e:
                        print(f"⚠️  Failed to apply {attention_config} to {workflow_name}: {e}")

            if not test_cases:
                pytest.skip("No test cases generated from config")

            # Parametrize the test
            metafunc.parametrize(
                "workflow_name,workflow,attention_config,test_id,timeout",
                test_cases,
                ids=[f"{wf}-{tid}" for wf, _, _, tid, _ in test_cases]
            )


if __name__ == "__main__":
    # Run tests with: pytest tests/test_workflows.py -v
    print("Run with: pytest tests/test_workflows.py -v -m meshcraft")
