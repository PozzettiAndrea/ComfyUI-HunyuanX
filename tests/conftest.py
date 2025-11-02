import os
import json
import pytest
from pathlib import Path

# Command line arguments for pytest
def pytest_addoption(parser):
    parser.addoption('--output_dir', action="store", default='output', help='Output directory for generated files')
    parser.addoption("--listen", type=str, default="127.0.0.1", metavar="IP", nargs="?", const="0.0.0.0", help="Specify the IP address to listen on (default: 127.0.0.1)")
    parser.addoption("--port", type=int, default=8188, help="Set the listen port.")
    parser.addoption("--workflow-dir", type=str, default=None, help="Directory containing workflow JSON files to test (defaults to ../workflows relative to tests)")
    parser.addoption("--test-results-dir", type=str, default="test_results", help="Directory to save test results and performance metrics")


# This initializes args at the beginning of the test session
@pytest.fixture(scope="session", autouse=True)
def args_pytest(pytestconfig):
    args = {}
    args['output_dir'] = pytestconfig.getoption('output_dir')
    args['listen'] = pytestconfig.getoption('listen')
    args['port'] = pytestconfig.getoption('port')

    os.makedirs(args['output_dir'], exist_ok=True)

    return args


# Workflow testing fixtures

@pytest.fixture(scope="session")
def workflow_dir(pytestconfig):
    """Get the directory containing workflow JSON files."""
    workflow_dir_str = pytestconfig.getoption("workflow_dir")

    if workflow_dir_str:
        return Path(workflow_dir_str)
    else:
        # Default to ../workflows relative to tests directory
        return Path(__file__).parent.parent / "workflows"


@pytest.fixture(scope="session")
def test_results_dir(pytestconfig):
    """Get the directory for saving test results."""
    results_dir_str = pytestconfig.getoption("test_results_dir")
    results_dir = Path(results_dir_str)
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


@pytest.fixture(scope="session")
def performance_tracker(test_results_dir):
    """Create a performance tracker for the test session."""
    from utils import PerformanceTracker
    return PerformanceTracker(output_dir=test_results_dir)


@pytest.fixture
def load_workflow():
    """Fixture factory for loading and converting workflow JSON files."""
    from utils import convert_workflow_file

    def _load_workflow(workflow_path):
        """
        Load a workflow JSON and convert to API format if needed.

        Args:
            workflow_path: Path to workflow JSON file (str or Path)

        Returns:
            Tuple of (workflow_dict, workflow_name)
        """
        workflow_path = Path(workflow_path)

        if not workflow_path.exists():
            raise FileNotFoundError(f"Workflow not found: {workflow_path}")

        # Convert to API format
        workflow_dict = convert_workflow_file(str(workflow_path))

        # Extract workflow name from filename
        workflow_name = workflow_path.stem

        return workflow_dict, workflow_name

    return _load_workflow


@pytest.fixture
def get_workflow_variants():
    """Fixture factory for generating workflow variants with different attention configs."""
    from utils import generate_workflow_variants, detect_workflow_model_type

    def _get_variants(workflow_dict):
        """
        Generate all attention configuration variants for a workflow.

        Args:
            workflow_dict: API-format workflow dict

        Returns:
            List of (AttentionConfig, modified_workflow) tuples
        """
        model_type = detect_workflow_model_type(workflow_dict)
        if model_type == "unknown":
            return []

        return generate_workflow_variants(workflow_dict, model_type)

    return _get_variants


@pytest.fixture(scope="session")
def meshcraft_workflows(workflow_dir):
    """
    Load all MeshCraft workflow JSON files.

    Returns:
        Dict mapping workflow names to file paths
    """
    if not workflow_dir.exists():
        pytest.skip(f"Workflow directory not found: {workflow_dir}")

    workflows = {}
    for json_file in workflow_dir.glob("*.json"):
        workflows[json_file.stem] = json_file

    if not workflows:
        pytest.skip(f"No workflow JSON files found in {workflow_dir}")

    return workflows


@pytest.fixture
def track_workflow_performance(performance_tracker):
    """
    Context manager fixture for tracking workflow execution performance.

    Usage:
        with track_workflow_performance(workflow_name, attention_config) as tracker:
            # Run workflow
            tracker.add_outputs(output_files)
    """
    import contextlib

    @contextlib.contextmanager
    def _track(workflow_name, attention_config, **metadata):
        """
        Track performance of workflow execution.

        Args:
            workflow_name: Name of the workflow
            attention_config: Attention configuration string
            **metadata: Additional metadata to track

        Yields:
            Tracker context with add_outputs method
        """
        class TrackerContext:
            def __init__(self, tracker):
                self.tracker = tracker
                self.output_files = []

            def add_outputs(self, files):
                """Add output files to the tracker."""
                if isinstance(files, (list, tuple)):
                    self.output_files.extend(files)
                else:
                    self.output_files.append(files)

        performance_tracker.start_test(
            workflow_name=workflow_name,
            attention_config=attention_config,
            **metadata
        )

        context = TrackerContext(performance_tracker)

        try:
            yield context
            performance_tracker.finish_test(
                status="success",
                output_files=context.output_files
            )
        except Exception as e:
            performance_tracker.finish_test(
                status="failed",
                error_message=str(e)
            )
            raise

    return _track
