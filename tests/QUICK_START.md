# Quick Start Guide - MeshCraft Workflow Testing

## ✅ Setup Complete!

All tests are now properly configured in the ComfyUI-MeshCraft repository.

---

## 🚀 Run Tests (3 Simple Steps)

### 1. Navigate to MeshCraft Directory
```bash
cd /workspace/Daxinzhuang-Pottery-Puzzle-Challenge/ComfyUI/custom_nodes/ComfyUI-MeshCraft
```

### 2. Verify Dependencies
```bash
# Make sure these are installed
pip install pytest websocket-client psutil pynvml
```

### 3. Run Tests!
```bash
# Run all 19 tests
./run_tests.sh test_workflows.py -v -m meshcraft

# Or just test one workflow
./run_tests.sh test_workflows.py -v -k "trellis-i2m"
```

---

## 📊 What Gets Tested

**18 workflow execution tests:**
- `trellis-i2m` with 8 attention configurations
- `trellis-t2m` with 8 attention configurations  
- `hunyuan-i2m` with 2 attention configurations

**Plus:**
- 1 final report generation test

**Total: 19 tests**

---

## 🎯 Common Commands

```bash
# Test specific attention backend
./run_tests.sh test_workflows.py -v -k "flash-attn"

# Test only Hunyuan workflows
./run_tests.sh test_workflows.py -v -k "hunyuan"

# Stop on first failure
./run_tests.sh test_workflows.py -v -m meshcraft -x

# Show full output
./run_tests.sh test_workflows.py -v -s -m meshcraft
```

---

## 📁 Test Results

After running tests, find results in:
```
test_results/
├── performance_metrics_YYYYMMDD_HHMMSS.json
└── performance_metrics_YYYYMMDD_HHMMSS.csv
```

Output files in:
```
output/
└── *.glb, *.png, etc.
```

---

## ⏱️ Expected Runtime

- **First Run:** ~20-30 minutes (downloads models)
- **Subsequent Runs:** ~15-20 minutes
- **Single Workflow:** ~5-8 minutes

---

## 🔍 Verify Setup

Check that tests are discoverable:
```bash
cd tests && pytest test_workflows.py --collect-only
```

Should show: `19 tests collected`

---

## 📚 More Help

- Full guide: `tests/README.md`
- Technical details: `tests/IMPLEMENTATION_SUMMARY.md`

---

**You're all set! Just run:**
```bash
./run_tests.sh test_workflows.py -v -m meshcraft
```
