#!/usr/bin/env python3
"""
Quick test script to verify Trimesh OSMesa rendering works in headless environment.
"""
import sys
from pathlib import Path

# Add tests directory to path
TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS_DIR))

from render_utils import render_glb_to_image, render_multiple_views
import trimesh


def test_basic_rendering():
    """Test rendering a simple cube mesh."""
    print("\n" + "="*60)
    print("TEST 1: Basic Cube Rendering")
    print("="*60)

    # Create a test mesh (cube)
    cube = trimesh.creation.box(extents=(1, 1, 1))

    # Save to temporary GLB file
    test_glb = TESTS_DIR / "test_cube.glb"
    cube.export(test_glb)
    print(f"✅ Created test cube mesh: {test_glb}")

    # Render it
    output_path = TESTS_DIR / "test_cube_render.png"
    success = render_glb_to_image(
        mesh_path=test_glb,
        output_path=output_path,
        image_size=(800, 800),
        camera_position="isometric",
        background_color="white"
    )

    if success and output_path.exists():
        print(f"✅ Successfully rendered cube to: {output_path}")
        print(f"   File size: {output_path.stat().st_size / 1024:.1f} KB")
        return True
    else:
        print(f"❌ Failed to render cube")
        return False


def test_multiple_views():
    """Test rendering multiple camera angles."""
    print("\n" + "="*60)
    print("TEST 2: Multiple View Rendering")
    print("="*60)

    # Create a test mesh (sphere)
    sphere = trimesh.creation.icosphere(subdivisions=2)

    # Save to temporary GLB file
    test_glb = TESTS_DIR / "test_sphere.glb"
    sphere.export(test_glb)
    print(f"✅ Created test sphere mesh: {test_glb}")

    # Render multiple views
    output_dir = TESTS_DIR / "test_renders"
    rendered_files = render_multiple_views(
        mesh_path=test_glb,
        output_dir=output_dir,
        views=["isometric", "front", "side", "top"],
        image_size=(400, 400)
    )

    if len(rendered_files) == 4:
        print(f"✅ Successfully rendered {len(rendered_files)} views")
        for f in rendered_files:
            print(f"   - {f.name}")
        return True
    else:
        print(f"❌ Expected 4 views, got {len(rendered_files)}")
        return False


def test_existing_glb():
    """Test rendering an existing GLB file if one exists."""
    print("\n" + "="*60)
    print("TEST 3: Rendering Existing GLB (if available)")
    print("="*60)

    # Look for GLB files in test output directories
    output_dir = TESTS_DIR / "output"
    glb_files = list(output_dir.rglob("*.glb"))

    if not glb_files:
        print("⚠️  No existing GLB files found in test outputs, skipping")
        return True

    # Use the first GLB file found
    test_glb = glb_files[0]
    print(f"📂 Found GLB file: {test_glb}")
    print(f"   Size: {test_glb.stat().st_size / 1024:.1f} KB")

    # Render it
    output_path = TESTS_DIR / f"render_{test_glb.stem}.png"
    success = render_glb_to_image(
        mesh_path=test_glb,
        output_path=output_path,
        image_size=(800, 800),
        camera_position="isometric",
        background_color="white"
    )

    if success and output_path.exists():
        print(f"✅ Successfully rendered GLB to: {output_path}")
        print(f"   File size: {output_path.stat().st_size / 1024:.1f} KB")
        return True
    else:
        print(f"❌ Failed to render GLB")
        return False


if __name__ == "__main__":
    print("\n🎨 Testing Trimesh OSMesa Rendering in Headless Environment")
    print("   This test should work WITHOUT X server or display")

    results = []

    # Run tests
    results.append(("Basic Rendering", test_basic_rendering()))
    results.append(("Multiple Views", test_multiple_views()))
    results.append(("Existing GLB", test_existing_glb()))

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status:10} {test_name}")

    total = len(results)
    passed = sum(1 for _, p in results if p)

    print(f"\nTotal: {passed}/{total} tests passed")

    # Exit with appropriate code
    sys.exit(0 if passed == total else 1)
