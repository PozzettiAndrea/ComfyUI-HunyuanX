# MeshCraft Workflow Automated Testing Guide

## Overview

This testing framework automatically tests ComfyUI-MeshCraft workflows with different attention configurations to validate functionality and measure performance.

### What Gets Tested

**Workflows:**
- `trellis-i2m.json` - Trellis image-to-3D workflow
- `trellis-t2m.json` - Trellis text-to-3D workflow
- `hunyuan-i2m.json` - Hunyuan3D image-to-3D workflow

**Attention Configurations:**

For **Trellis** workflows (8 combinations):
- `attn_backend`: flash-attn, sdpa, naive, xformers
- `spconv_algo`: auto, flash-native

For **Hunyuan** workflows (2 combinations):
- `attention_mode`: sdpa, sageattn

**Total Test Cases: 18**
- trellis-i2m: 8 tests
- trellis-t2m: 8 tests
- hunyuan-i2m: 2 tests

### What Gets Measured

- ✅ Workflow execution success/failure
- ⏱️  Execution time per configuration
- 💾 GPU memory usage (if available)
- 📦 Output file generation
- 📊 Performance comparison across configurations

---

## Prerequisites

### 1. Install Test Dependencies

```bash
pip install pytest websocket-client psutil pynvml
```

### 2. Verify Workflow Files Exist

```bash
ls workflows/
# Should show: trellis-i2m.json, trellis-t2m.json, hunyuan-i2m.json
```

### 3. Ensure Required Models Are Downloaded

The workflows will attempt to download models automatically, but you can pre-download them:

- **Trellis models:** Will auto-download on first run
- **Hunyuan3D models:** Will auto-download on first run
- **DinoV2 models:** Will auto-download on first run

---

## Running Tests

### Quick Start - Run All Tests

```bash
# From ComfyUI-MeshCraft directory
cd /workspace/Daxinzhuang-Pottery-Puzzle-Challenge/ComfyUI/custom_nodes/ComfyUI-MeshCraft

# Run all MeshCraft workflow tests
pytest tests/test_workflows.py -v -m meshcraft
```

### Run Specific Workflows

```bash
# Test only trellis-i2m workflow
pytest tests/test_workflows.py -v -k "trellis-i2m"

# Test only hunyuan workflows
pytest tests/test_workflows.py -v -k "hunyuan"

# Test only flash-attn configurations
pytest tests/test_workflows.py -v -k "flash-attn"
```

### Run with Custom Settings

```bash
# Custom output directory
pytest tests/test_workflows.py -v -m meshcraft \
  --output_dir=/tmp/comfy_test_outputs

# Custom workflow directory
pytest tests/test_workflows.py -v -m meshcraft \
  --workflow-dir=../workflows

# Custom test results directory
pytest tests/test_workflows.py -v -m meshcraft \
  --test-results-dir=my_test_results
```

### Run with Different Server Settings

```bash
# Use custom port
pytest tests/test_meshcraft_workflows.py -v -m meshcraft --port 8189

# Listen on all interfaces
pytest tests/test_meshcraft_workflows.py -v -m meshcraft --listen 0.0.0.0
```

---

## Test Output

### Console Output

During test execution, you'll see:

```
================================ test session starts =================================
collecting ... collected 18 items

tests/test_meshcraft_workflows.py::TestMeshCraftWorkflows::test_workflow_with_attention_config[trellis-i2m-flash-attn-auto]
============================================================
Testing: trellis-i2m with flash-attn_auto
============================================================
✅ Success! Generated 3 outputs
   Outputs: ['output_17186.glb', ...]
PASSED

tests/test_meshcraft_workflows.py::TestMeshCraftWorkflows::test_workflow_with_attention_config[trellis-i2m-flash-attn-flash-native]
...
```

### Performance Report

At the end of the test run, a summary is printed:

```
============================================================
PERFORMANCE TEST SUMMARY
============================================================
Total Tests: 18
Successful: 16
Failed: 2
Success Rate: 88.9%

Timing:
  Average Duration: 45.2s
  Min Duration: 32.1s
  Max Duration: 67.8s

GPU Memory:
  Average Usage: 12543 MB
  Peak Usage: 15234 MB

By Workflow:
  trellis-i2m: 7/8 passed
  trellis-t2m: 7/8 passed
  hunyuan-i2m: 2/2 passed
============================================================
```

### Generated Files

After testing, you'll find:

1. **Test Results:**
   ```
   test_results/
   ├── performance_metrics_YYYYMMDD_HHMMSS.json
   └── performance_metrics_YYYYMMDD_HHMMSS.csv
   ```

2. **Output Files:**
   ```
   tests/inference/samples/
   ├── output_xxxxx.glb
   ├── output_xxxxx.png
   └── ...
   ```

---

## Understanding Test Results

### CSV Report Format

The CSV report contains one row per test with columns:

| Column | Description |
|--------|-------------|
| `workflow_name` | Name of the workflow (e.g., "trellis-i2m") |
| `attention_config` | Attention configuration used |
| `test_id` | Unique test identifier |
| `start_time` | Test start timestamp (ISO 8601) |
| `end_time` | Test end timestamp |
| `duration_seconds` | Execution time in seconds |
| `status` | "success" or "failed" |
| `error_message` | Error details if failed |
| `gpu_memory_mb` | Peak GPU memory usage (MB) |
| `cpu_memory_mb` | CPU memory usage (MB) |
| `num_outputs` | Number of output files generated |
| `model_type` | "trellis" or "hunyuan" |

### JSON Report Format

```json
{
  "timestamp": "2025-11-02T12:00:00",
  "total_tests": 18,
  "successful_tests": 16,
  "failed_tests": 2,
  "metrics": [
    {
      "workflow_name": "trellis-i2m",
      "attention_config": "flash-attn_flash-native",
      "test_id": "trellis-flash-attn-flash-native",
      "start_time": "2025-11-02T12:00:10",
      "end_time": "2025-11-02T12:00:55",
      "duration_seconds": 45.2,
      "status": "success",
      "gpu_memory_mb": 12543.5,
      "num_outputs": 3,
      ...
    },
    ...
  ]
}
```

---

## Analyzing Results

### Find Fastest Configuration

```bash
# Sort CSV by duration
sort -t, -k6 -n test_results/performance_metrics_*.csv | head -5
```

### Find Configuration with Lowest Memory Usage

```bash
# Sort by GPU memory
sort -t, -k9 -n test_results/performance_metrics_*.csv | grep success | head -5
```

### Compare Attention Backends

```python
import pandas as pd

# Load results
df = pd.read_csv('test_results/performance_metrics_YYYYMMDD_HHMMSS.csv')

# Group by attention config
summary = df.groupby('attention_config').agg({
    'duration_seconds': ['mean', 'std'],
    'gpu_memory_mb': ['mean', 'max'],
    'status': lambda x: (x == 'success').sum()
})

print(summary)
```

---

## Troubleshooting

### Issue: Tests Hang or Timeout

**Cause:** Workflow execution taking longer than 600s timeout

**Solution:**
```bash
# Increase timeout in test_meshcraft_workflows.py
# Edit line: result = client.execute_workflow(workflow, timeout=1200)
```

### Issue: CUDA Out of Memory

**Cause:** GPU VRAM exhausted

**Solutions:**
1. Run tests one at a time:
   ```bash
   pytest tests/test_meshcraft_workflows.py -v -k "trellis-i2m-flash-attn-auto"
   ```

2. Reduce batch size in workflows (if applicable)

3. Use CPU-only mode:
   ```bash
   pytest tests/test_meshcraft_workflows.py -v --cpu
   ```

### Issue: Connection Refused to Server

**Cause:** Server not starting or port already in use

**Solutions:**
1. Check if port 8188 is already in use:
   ```bash
   lsof -i :8188
   ```

2. Use a different port:
   ```bash
   pytest tests/test_meshcraft_workflows.py --port 8189
   ```

3. Increase connection retry count in test

### Issue: Workflow Not Found

**Cause:** Workflow file missing or wrong directory

**Solution:**
```bash
# Verify workflow directory
ls custom_nodes/ComfyUI-MeshCraft/workflows/

# Or specify custom directory
pytest tests/test_meshcraft_workflows.py \
  --workflow-dir=/path/to/workflows
```

### Issue: Model Download Failures

**Cause:** Network issues or insufficient disk space

**Solutions:**
1. Pre-download models manually
2. Check disk space: `df -h`
3. Check internet connection

---

## Advanced Usage

### Testing Custom Workflows

1. Place your workflow JSON in the workflows directory:
   ```bash
   cp my_workflow.json custom_nodes/ComfyUI-MeshCraft/workflows/
   ```

2. Add workflow name to `TARGET_WORKFLOWS` in `test_meshcraft_workflows.py`:
   ```python
   TARGET_WORKFLOWS = ["trellis-i2m", "trellis-t2m", "hunyuan-i2m", "my_workflow"]
   ```

3. Run tests:
   ```bash
   pytest tests/test_meshcraft_workflows.py -v -k "my_workflow"
   ```

### Integrating with CI/CD

Add to your GitHub Actions workflow:

```yaml
name: Test MeshCraft Workflows

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v2

      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest websocket-client psutil

      - name: Run MeshCraft workflow tests
        run: |
          cd ComfyUI
          pytest tests/test_meshcraft_workflows.py -v -m meshcraft

      - name: Upload test results
        uses: actions/upload-artifact@v2
        with:
          name: test-results
          path: test_results/
```

---

## Utility Scripts

### Standalone Workflow Converter

Convert a workflow file from UI format to API format:

```bash
python tests/utils/workflow_converter.py \
  custom_nodes/ComfyUI-MeshCraft/workflows/trellis-i2m.json \
  /tmp/trellis-i2m-api.json
```

### List Available Attention Configs

```bash
python tests/utils/attention_configs.py
```

Output:
```
=== Trellis Attention Configurations ===
  flash-attn_auto: {'attn_backend': 'flash-attn', 'spconv_algo': 'auto'}
  flash-attn_flash-native: {'attn_backend': 'flash-attn', 'spconv_algo': 'flash-native'}
  ...

Total Trellis configs: 8

=== Hunyuan Attention Configurations ===
  sdpa: {'attention_mode': 'sdpa'}
  sageattn: {'attention_mode': 'sageattn'}

Total Hunyuan configs: 2
```

### Generate Performance Report

```bash
python tests/utils/performance_tracker.py
```

---

## Architecture

### Components

```
tests/
├── test_meshcraft_workflows.py   # Main test suite
├── conftest.py                    # Pytest fixtures and configuration
├── utils/
│   ├── workflow_converter.py     # UI→API workflow conversion
│   ├── attention_configs.py      # Attention config management
│   └── performance_tracker.py    # Performance metrics tracking
└── MESHCRAFT_TESTING_GUIDE.md    # This guide
```

### Test Flow

```
1. pytest collects tests
   ↓
2. pytest_generate_tests() creates 18 test cases
   ↓
3. Server starts (via _server fixture)
   ↓
4. Client connects (via client fixture)
   ↓
5. For each test case:
   a. Load workflow
   b. Apply attention config
   c. Start performance tracking
   d. Execute workflow
   e. Validate outputs
   f. Record metrics
   ↓
6. Server stops
   ↓
7. Generate final report
```

---

## FAQ

**Q: How long does the full test suite take?**
A: Approximately 15-30 minutes depending on GPU speed and whether models need to be downloaded.

**Q: Can I run tests in parallel?**
A: Not recommended - workflows share GPU memory and may interfere with each other.

**Q: How much disk space is needed?**
A: ~50GB for models + ~5GB for test outputs.

**Q: Can I test on CPU only?**
A: Yes, but it will be much slower (2-5x).

**Q: What if a test fails?**
A: Check the error message in the console output and the CSV report's `error_message` column. Common issues are CUDA OOM, missing models, or workflow syntax errors.

**Q: How do I add a new attention configuration?**
A: Edit `tests/utils/attention_configs.py` and add to the `TRELLIS_ATTN_BACKENDS`, `TRELLIS_SPCONV_ALGOS`, or `HUNYUAN_ATTENTION_MODES` lists.

---

## Support

For issues or questions:
1. Check the Troubleshooting section above
2. Review test output and error messages
3. Check ComfyUI-MeshCraft documentation
4. Open an issue on GitHub

---

## License

Same as ComfyUI and ComfyUI-MeshCraft
