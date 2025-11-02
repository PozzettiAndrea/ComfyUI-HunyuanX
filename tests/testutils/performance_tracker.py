"""
Performance tracking utility for ComfyUI workflow tests.

Tracks execution time, memory usage, and test results for workflow testing.
"""

import time
import json
import csv
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict, field
from datetime import datetime


@dataclass
class PerformanceMetrics:
    """Performance metrics for a single workflow execution."""

    # Test identification
    workflow_name: str
    attention_config: str
    test_id: str

    # Timing
    start_time: str
    end_time: str = None
    duration_seconds: float = None

    # Status
    status: str = "pending"  # pending, running, success, failed
    error_message: str = None

    # Resource usage
    gpu_memory_mb: float = None
    cpu_memory_mb: float = None

    # Outputs
    output_files: List[str] = field(default_factory=list)
    num_outputs: int = 0

    # Metadata
    model_type: str = None
    node_count: int = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class PerformanceTracker:
    """Tracks performance metrics for workflow testing."""

    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize performance tracker.

        Args:
            output_dir: Directory to save performance reports (default: ./test_results)
        """
        self.output_dir = Path(output_dir) if output_dir else Path("test_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.metrics: List[PerformanceMetrics] = []
        self.current_metric: Optional[PerformanceMetrics] = None

        # Try to import GPU monitoring libraries
        self.gpu_available = self._check_gpu_availability()

    def _check_gpu_availability(self) -> bool:
        """Check if GPU monitoring is available."""
        try:
            import pynvml
            pynvml.nvmlInit()
            return True
        except (ImportError, Exception):
            return False

    def _get_gpu_memory_mb(self) -> Optional[float]:
        """Get current GPU memory usage in MB."""
        if not self.gpu_available:
            return None

        try:
            import pynvml
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return mem_info.used / (1024 ** 2)  # Convert bytes to MB
        except Exception:
            return None

    def _get_cpu_memory_mb(self) -> Optional[float]:
        """Get current CPU memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024 ** 2)  # Convert bytes to MB
        except (ImportError, Exception):
            return None

    def start_test(
        self,
        workflow_name: str,
        attention_config: str,
        test_id: str = None,
        model_type: str = None,
        node_count: int = None,
        **metadata
    ) -> None:
        """
        Start tracking a new test execution.

        Args:
            workflow_name: Name of the workflow being tested
            attention_config: Attention configuration being used
            test_id: Optional unique test ID
            model_type: Type of model (trellis/hunyuan)
            node_count: Number of nodes in the workflow
            **metadata: Additional metadata to store
        """
        if self.current_metric is not None:
            # Auto-finish previous test if not finished
            self.finish_test(status="abandoned")

        test_id = test_id or f"{workflow_name}_{attention_config}_{int(time.time())}"

        self.current_metric = PerformanceMetrics(
            workflow_name=workflow_name,
            attention_config=attention_config,
            test_id=test_id,
            start_time=datetime.now().isoformat(),
            status="running",
            model_type=model_type,
            node_count=node_count,
            metadata=metadata
        )

        # Record initial memory
        self.current_metric.gpu_memory_mb = self._get_gpu_memory_mb()
        self.current_metric.cpu_memory_mb = self._get_cpu_memory_mb()

    def finish_test(
        self,
        status: str = "success",
        error_message: str = None,
        output_files: List[str] = None
    ) -> PerformanceMetrics:
        """
        Finish tracking the current test execution.

        Args:
            status: Test status ("success", "failed", "timeout", etc.)
            error_message: Error message if test failed
            output_files: List of output file paths generated

        Returns:
            PerformanceMetrics object for the finished test
        """
        if self.current_metric is None:
            raise ValueError("No active test to finish")

        self.current_metric.end_time = datetime.now().isoformat()
        self.current_metric.status = status
        self.current_metric.error_message = error_message

        if output_files:
            self.current_metric.output_files = output_files
            self.current_metric.num_outputs = len(output_files)

        # Calculate duration
        start = datetime.fromisoformat(self.current_metric.start_time)
        end = datetime.fromisoformat(self.current_metric.end_time)
        self.current_metric.duration_seconds = (end - start).total_seconds()

        # Update memory measurements (peak usage)
        gpu_mem = self._get_gpu_memory_mb()
        if gpu_mem and (self.current_metric.gpu_memory_mb is None or gpu_mem > self.current_metric.gpu_memory_mb):
            self.current_metric.gpu_memory_mb = gpu_mem

        cpu_mem = self._get_cpu_memory_mb()
        if cpu_mem and (self.current_metric.cpu_memory_mb is None or cpu_mem > self.current_metric.cpu_memory_mb):
            self.current_metric.cpu_memory_mb = cpu_mem

        # Store in history
        self.metrics.append(self.current_metric)
        finished_metric = self.current_metric
        self.current_metric = None

        return finished_metric

    def save_results(self, format: str = "both") -> Dict[str, Path]:
        """
        Save performance results to file(s).

        Args:
            format: Output format ("json", "csv", or "both")

        Returns:
            Dict mapping format names to file paths
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_files = {}

        if format in ["json", "both"]:
            json_path = self.output_dir / f"performance_metrics_{timestamp}.json"
            self._save_json(json_path)
            output_files["json"] = json_path

        if format in ["csv", "both"]:
            csv_path = self.output_dir / f"performance_metrics_{timestamp}.csv"
            self._save_csv(csv_path)
            output_files["csv"] = csv_path

        return output_files

    def _save_json(self, path: Path) -> None:
        """Save results as JSON."""
        data = {
            "timestamp": datetime.now().isoformat(),
            "total_tests": len(self.metrics),
            "successful_tests": sum(1 for m in self.metrics if m.status == "success"),
            "failed_tests": sum(1 for m in self.metrics if m.status == "failed"),
            "metrics": [asdict(m) for m in self.metrics]
        }

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    def _save_csv(self, path: Path) -> None:
        """Save results as CSV."""
        if not self.metrics:
            return

        # Flatten the dataclass for CSV (exclude complex fields)
        fieldnames = [
            'workflow_name', 'attention_config', 'test_id',
            'start_time', 'end_time', 'duration_seconds',
            'status', 'error_message',
            'gpu_memory_mb', 'cpu_memory_mb',
            'num_outputs', 'model_type', 'node_count'
        ]

        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for metric in self.metrics:
                row = asdict(metric)
                # Remove complex fields
                row.pop('output_files', None)
                row.pop('metadata', None)
                # Keep only fields in fieldnames
                row = {k: v for k, v in row.items() if k in fieldnames}
                writer.writerow(row)

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all tracked metrics.

        Returns:
            Dict with summary statistics
        """
        if not self.metrics:
            return {"total_tests": 0}

        successful = [m for m in self.metrics if m.status == "success"]
        failed = [m for m in self.metrics if m.status == "failed"]

        summary = {
            "total_tests": len(self.metrics),
            "successful_tests": len(successful),
            "failed_tests": len(failed),
            "success_rate": len(successful) / len(self.metrics) if self.metrics else 0,
        }

        if successful:
            durations = [m.duration_seconds for m in successful if m.duration_seconds]
            if durations:
                summary["avg_duration_seconds"] = sum(durations) / len(durations)
                summary["min_duration_seconds"] = min(durations)
                summary["max_duration_seconds"] = max(durations)

            gpu_mems = [m.gpu_memory_mb for m in successful if m.gpu_memory_mb]
            if gpu_mems:
                summary["avg_gpu_memory_mb"] = sum(gpu_mems) / len(gpu_mems)
                summary["max_gpu_memory_mb"] = max(gpu_mems)

        # Group by workflow
        by_workflow = {}
        for metric in self.metrics:
            wf_name = metric.workflow_name
            if wf_name not in by_workflow:
                by_workflow[wf_name] = {"total": 0, "success": 0, "failed": 0}
            by_workflow[wf_name]["total"] += 1
            if metric.status == "success":
                by_workflow[wf_name]["success"] += 1
            elif metric.status == "failed":
                by_workflow[wf_name]["failed"] += 1

        summary["by_workflow"] = by_workflow

        return summary

    def print_summary(self) -> None:
        """Print a formatted summary of results."""
        summary = self.get_summary()

        print("\n" + "=" * 60)
        print("PERFORMANCE TEST SUMMARY")
        print("=" * 60)
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Successful: {summary.get('successful_tests', 0)}")
        print(f"Failed: {summary.get('failed_tests', 0)}")
        print(f"Success Rate: {summary.get('success_rate', 0):.1%}")

        if 'avg_duration_seconds' in summary:
            print(f"\nTiming:")
            print(f"  Average Duration: {summary['avg_duration_seconds']:.2f}s")
            print(f"  Min Duration: {summary['min_duration_seconds']:.2f}s")
            print(f"  Max Duration: {summary['max_duration_seconds']:.2f}s")

        if 'avg_gpu_memory_mb' in summary:
            print(f"\nGPU Memory:")
            print(f"  Average Usage: {summary['avg_gpu_memory_mb']:.0f} MB")
            print(f"  Peak Usage: {summary['max_gpu_memory_mb']:.0f} MB")

        if 'by_workflow' in summary and summary['by_workflow']:
            print(f"\nBy Workflow:")
            for wf_name, stats in summary['by_workflow'].items():
                print(f"  {wf_name}: {stats['success']}/{stats['total']} passed")

        print("=" * 60 + "\n")


if __name__ == "__main__":
    # Simple test
    tracker = PerformanceTracker(output_dir=Path("/tmp/comfy_test_results"))

    # Simulate some tests
    tracker.start_test("trellis-i2m", "flash-attn_flash-native", model_type="trellis")
    time.sleep(0.1)
    tracker.finish_test(status="success", output_files=["output1.glb"])

    tracker.start_test("trellis-i2m", "sdpa_auto", model_type="trellis")
    time.sleep(0.05)
    tracker.finish_test(status="success", output_files=["output2.glb"])

    tracker.start_test("hunyuan-i2m", "sdpa", model_type="hunyuan")
    time.sleep(0.08)
    tracker.finish_test(status="failed", error_message="CUDA out of memory")

    # Print summary
    tracker.print_summary()

    # Save results
    files = tracker.save_results()
    print(f"Results saved to:")
    for fmt, path in files.items():
        print(f"  {fmt}: {path}")
