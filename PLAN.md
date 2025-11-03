# ComfyUI-MeshCraft Workflow Testing Plan

**Date:** 2025-10-31
**Status:** In Progress - Workflow Conversion Research Phase

---

## 🎯 Original Goal

Add automated workflow tests that:
1. Load saved workflow JSON files (trellis-i2m.json, etc.)
2. Execute workflows with different attention mechanism combinations
3. Validate outputs to catch regressions

---

## ✅ Completed Tasks

### 1. Fixed TRELLIS Import Error
- **Problem:** Missing `lib/trellis/models/` directory causing import failure
- **Solution:** Downloaded 11 missing Python files from Microsoft TRELLIS repo
- **Files added:**
  - `lib/trellis/models/__init__.py`
  - `lib/trellis/models/*.py` (5 files)
  - `lib/trellis/models/structured_latent_vae/*.py` (6 files)

### 2. Fixed hy3dshape Import Error
- **Problem:** Missing `lib/hy3dshape/hy3dshape/models/` directory
- **Solution:** Restored from git history (commit `417047a`)
- **Files restored:** 18 Python files across 4 subdirectories

### 3. Updated .gitignore
- **Problem:** Model directories were being ignored, including source code
- **Solution:** Updated rules to:
  ```gitignore
  models/                    # Ignore all models directories
  !lib/trellis/models/       # Except TRELLIS source code
  !lib/hy3dshape/hy3dshape/models/  # Except hy3dshape source code

  # Ignore weight files everywhere
  *.pth, *.safetensors, *.ckpt, etc.
  *.npz  # Training data
  ```

### 4. Modified GitHub Actions Workflow
- **File:** `.github/workflows/gpu-test.yml`
- **Change:** Removed `push: branches: [main]` trigger
- **Result:** Tests only run on manual trigger (`workflow_dispatch`)

### 5. Created Workflow Test Infrastructure

**New Files:**
- `tests/test_workflows.py` (319 lines)
  - 13 tests collected
  - 4 test classes: Structure, NodeConfiguration, AttentionMechanisms, WorkflowExecution
  - Parametrized tests for attention variants
  - Proper markers: `@pytest.mark.workflow`, `@pytest.mark.gpu`, `@pytest.mark.slow`

- Updated `tests/conftest.py` (+85 lines)
  - `workflow_test_image()` fixture - Creates test image
  - `load_workflow_json()` fixture - Loads workflow files
  - `modify_workflow_attention()` fixture - Modifies attention settings

- Updated `pytest.ini`
  - Added `workflow` marker

**Tests Status:**
- ✅ Structure tests work (validate JSON, check required nodes)
- ✅ Attention modification tests work (parametrized across variants)
- ⏸️ Execution tests skipped (need workflow converter first)

---

## 🔍 Key Discovery: Workflow Format Conversion

### The Problem

ComfyUI has **TWO workflow formats:**

#### 1. **UI Format** (what we have)
```json
{
  "nodes": [{
    "id": 50,
    "type": "PreviewImage",
    "inputs": [{"name": "images", "link": 17}],
    "widgets_values": []
  }],
  "links": [
    [17, 45, 0, 50, 0, "IMAGE"]  // [link_id, src_node, src_slot, tgt_node, tgt_slot, type]
  ]
}
```

#### 2. **API Format** (what execution needs)
```json
{
  "50": {
    "inputs": {
      "images": ["45", 0]  // [source_node_id, output_index]
    },
    "class_type": "PreviewImage"
  }
}
```

### The Conversion Challenge

**Frontend does it:** ComfyUI's web UI has `app.graphToPrompt()` in JavaScript that converts UI → API format when you click "Queue Prompt"

**We need Python version:** For automated testing, we need to convert workflows in Python

---

## 🚀 Three Options Going Forward

### Option 1: Save Workflows in API Format (EASIEST) ✅ RECOMMENDED

**How:**
1. In ComfyUI web UI: Settings → Enable "Dev Mode"
2. Click "Save (API Format)" button instead of regular save
3. Saves as `workflow_api.json`

**Pros:**
- Zero code needed
- Official format
- Immediate solution

**Cons:**
- Need to maintain both UI and API versions
- Lose visual layout if you want to edit

**Implementation:**
```python
# No conversion needed!
with open("workflows/trellis-i2m_api.json") as f:
    api_workflow = json.load(f)

# Execute directly
result = execute_workflow(api_workflow)
```

---

### Option 2: Install Converter Custom Node

**Repository:** https://github.com/SethRobinson/comfyui-workflow-to-api-converter-endpoint

**How:**
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/SethRobinson/comfyui-workflow-to-api-converter-endpoint
# Restart ComfyUI
```

**Usage:**
```python
import requests

# Start ComfyUI server first
with open("workflows/trellis-i2m.json") as f:
    ui_workflow = json.load(f)

response = requests.post(
    "http://localhost:8188/workflow/convert",
    json=ui_workflow
)
api_workflow = response.json()
```

**Pros:**
- Uses official conversion logic (ported from JS)
- Works with existing UI workflows
- Maintained by community

**Cons:**
- Requires ComfyUI server running
- External dependency
- Adds latency to tests

---

### Option 3: Port JavaScript `app.graphToPrompt()` to Python

**Status:** 🔴 Most Complex - NOT RECOMMENDED

**Why it's hard:**
1. Frontend code is minified/compiled TypeScript
2. Widget mapping is complex (need to query NODE_CLASS_MAPPINGS)
3. Link resolution requires graph traversal
4. Edge cases around different input types

**Research findings:**
- Found the JS function in compiled frontend
- ~200 lines of complex graph traversal
- Relies on browser-side state
- Would require significant porting effort

**Estimated effort:** 4-6 hours of development + testing

---

## 📋 Recommended Next Steps

### Immediate (Next Session):

1. **Choose Option 1 or 2**
   - If quick solution: Use Option 1 (save API format workflows)
   - If want automation: Use Option 2 (converter custom node)

2. **If Option 1 (RECOMMENDED):**
   ```bash
   # In ComfyUI web UI:
   # 1. Load workflows/trellis-i2m.json
   # 2. Enable Dev Mode in settings
   # 3. Click "Save (API Format)"
   # 4. Save as workflows/api/trellis-i2m.json

   # Repeat for all workflows:
   - trellis-i2m.json → api/trellis-i2m.json
   - trellis-t2m.json → api/trellis-t2m.json
   - hunyuan-i2m.json → api/hunyuan-i2m.json
   ```

3. **Update test_workflows.py:**
   ```python
   # Change load path
   def load_workflow(name):
       api_path = f"workflows/api/{name}"
       with open(api_path) as f:
           return json.load(f)
   ```

4. **If Option 2:**
   ```bash
   # Install converter
   cd ComfyUI/custom_nodes
   git clone https://github.com/SethRobinson/comfyui-workflow-to-api-converter-endpoint

   # Restart ComfyUI
   cd ../..
   python main.py
   ```

5. **Add Workflow Execution Tests**
   - Create `execute_workflow()` helper in conftest.py
   - Use PromptExecutor + MockServer pattern
   - Un-skip execution tests in test_workflows.py
   - Add output validation (GLB file exists, size > 0)

### Medium Term:

1. **Add test images** to `tests/test_data/`
2. **Create manual test script** for debugging
3. **Add output validation** (mesh quality checks)
4. **Document test patterns** in TESTING.md

### Long Term:

1. **Add performance benchmarks** (execution time per attention variant)
2. **Add visual regression tests** (compare rendered outputs)
3. **Integrate with CI/CD** (auto-run on PRs)

---

## 📂 Current File Structure

```
ComfyUI-MeshCraft/
├── .github/workflows/
│   └── gpu-test.yml                    # ✅ Updated (manual trigger only)
├── tests/
│   ├── conftest.py                     # ✅ Updated (workflow fixtures)
│   ├── test_workflows.py               # ✅ New (13 tests, 7 passing)
│   ├── test_basic_nodes.py             # ✅ Existing
│   └── test_fixtures.py                # ✅ Existing
├── workflows/
│   ├── trellis-i2m.json                # ✅ UI format
│   ├── trellis-t2m.json                # ✅ UI format
│   ├── hunyuan-i2m.json                # ✅ UI format
│   └── api/                            # 🔲 TODO: Create API format versions
│       ├── trellis-i2m.json            # 🔲 TODO
│       ├── trellis-t2m.json            # 🔲 TODO
│       └── hunyuan-i2m.json            # 🔲 TODO
├── lib/
│   ├── trellis/
│   │   └── models/                     # ✅ Fixed (11 files downloaded)
│   └── hy3dshape/hy3dshape/
│       └── models/                     # ✅ Fixed (restored from git)
├── pytest.ini                          # ✅ Updated (workflow marker)
├── .gitignore                          # ✅ Updated (model exclusions)
└── PLAN.md                             # ✅ This file
```

---

## 🐛 Known Issues

1. **Test execution from wrong directory**
   - Tests must run from ComfyUI root with proper PYTHONPATH
   - Command: `cd /workspace/.../ComfyUI && PYTHONPATH=$PWD pytest custom_nodes/ComfyUI-MeshCraft/tests/`

2. **Missing test images**
   - Workflow execution tests need actual images
   - Available: `ComfyUI/input/example.png`, `typical_building_castle.png`

3. **No actual execution yet**
   - Execution tests are skipped (`@pytest.mark.skip`)
   - Need workflow converter before enabling

---

## 📚 Research Links

- **SethRobinson's Converter:** https://github.com/SethRobinson/comfyui-workflow-to-api-converter-endpoint
- **ComfyUI Issues:**
  - API conversion: https://github.com/comfyanonymous/ComfyUI/issues/1112
  - Format confusion: https://github.com/comfyanonymous/ComfyUI/issues/1335
- **Guides:**
  - ViewComfy integration: https://www.viewcomfy.com/blog/integrate-comfyui-workflows-into-your-apps-via-api
  - Modal.com guide: https://modal.com/blog/comfyui-prototype-to-production

---

## 💡 Key Insights

1. **ComfyUI already solves this** - Don't reinvent the wheel
2. **Frontend JS conversion** is complex and unnecessary
3. **API format workflows** are production-ready
4. **Testing infrastructure** is solid, just needs execution
5. **Attention mechanism testing** is well-designed and ready

---

## 🎬 Next Session Checklist

- [ ] Decide: Option 1 (API format) or Option 2 (converter node)?
- [ ] If Option 1: Save workflows in API format
- [ ] If Option 2: Install converter custom node
- [ ] Update `test_workflows.py` to load correct format
- [ ] Implement `execute_workflow()` helper
- [ ] Un-skip execution tests
- [ ] Run first full workflow test
- [ ] Add output validation
- [ ] Document testing patterns

---

**Last Updated:** 2025-10-31
**Next Review:** When starting workflow execution implementation
