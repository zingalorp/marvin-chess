#!/usr/bin/env python3
"""
Build standalone executable for Marvin ONNX UCI Engine using PyInstaller.

This creates a distributable folder with:
- marvin-onnx.exe (Windows) or marvin-onnx (Linux/macOS)
- All required DLLs/libraries
- ONNX model files

Usage:
    python scripts/build_executable.py

Requirements:
    pip install pyinstaller
    pip install onnxruntime>=1.17.0,<1.19.0
    pip install numpy>=1.24.0,<2.0.0
    
Note: onnxruntime 1.18.x works best with PyInstaller. Later versions (1.20+)
have DLL loading issues when frozen.

The output will be in dist/marvin-onnx/
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def check_pyinstaller():
    """Check if PyInstaller is installed."""
    try:
        import PyInstaller
        return True
    except ImportError:
        return False


def check_onnx_model(repo_root: Path) -> bool:
    """Check if ONNX model exists."""
    model_path = repo_root / "inference" / "marvin_small.onnx"
    data_path = repo_root / "inference" / "marvin_small.onnx.data"
    
    if not model_path.exists():
        print(f"ONNX model not found: {model_path}")
        print("  Run 'python scripts/export_onnx.py' first.")
        return False
    
    if not data_path.exists():
        print(f"ONNX data file not found: {data_path}")
        return False
    
    print(f"Found ONNX model ({model_path.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"Found ONNX data ({data_path.stat().st_size / 1024 / 1024:.1f} MB)")
    return True


def build_executable(repo_root: Path, use_onefile: bool = False) -> Path:
    """Build the executable using PyInstaller."""
    
    # Use direct PyInstaller invocation instead of spec file for reliability
    uci_script = repo_root / "inference" / "uci_onnx.py"
    onnx_model = repo_root / "inference" / "marvin_small.onnx"
    onnx_data = repo_root / "inference" / "marvin_small.onnx.data"
    
    if not uci_script.exists():
        print(f"UCI script not found: {uci_script}")
        sys.exit(1)
    
    # Clean previous builds
    dist_dir = repo_root / "dist" / "marvin-onnx"
    
    if dist_dir.exists():
        print(f"Cleaning previous build: {dist_dir}")
        shutil.rmtree(dist_dir)
    
    # Run PyInstaller
    print("\nBuilding executable with PyInstaller...")
    print("   This may take a few minutes...\n")
    
    # Path to custom hook
    hooks_dir = repo_root / "scripts"
    
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--clean",
        "--noconfirm",
        "--name", "marvin-onnx",
        "--console",  # UCI engines need console
        f"--add-data={onnx_model}{os.pathsep}.",
        f"--add-data={onnx_data}{os.pathsep}.",
        # Use custom hooks directory
        f"--additional-hooks-dir={hooks_dir}",
        # Collect all onnxruntime files to ensure DLLs are included
        "--collect-all=onnxruntime",
        "--collect-binaries=onnxruntime",
        "--hidden-import=onnxruntime",
        "--hidden-import=onnxruntime.capi",
        "--hidden-import=onnxruntime.capi._pybind_state",
        "--hidden-import=numpy",
        "--hidden-import=chess",
        # Exclude heavy packages we don't need
        "--exclude-module=torch",
        "--exclude-module=torchvision",
        "--exclude-module=tensorflow",
        "--exclude-module=matplotlib",
        "--exclude-module=scipy",
        "--exclude-module=pandas",
        "--exclude-module=PIL",
        str(uci_script),
    ]
    
    result = subprocess.run(cmd, cwd=str(repo_root))
    
    if result.returncode != 0:
        print("\nPyInstaller build failed!")
        sys.exit(1)
    
    # Check output
    if sys.platform == "win32":
        exe_path = dist_dir / "marvin-onnx.exe"
    else:
        exe_path = dist_dir / "marvin-onnx"
    
    if not exe_path.exists():
        print(f"\nExecutable not found: {exe_path}")
        sys.exit(1)
    
    # Copy launcher scripts
    release_dir = repo_root / "release"
    if (release_dir / "README.md").exists():
        shutil.copy(release_dir / "README.md", dist_dir / "README.md")
    
    # Create a simple batch file for Windows users
    if sys.platform == "win32":
        bat_content = '@echo off\n"%~dp0marvin-onnx.exe" %*\n'
        (dist_dir / "run.bat").write_text(bat_content)
        
        # Copy onnxruntime DLLs to _internal root where other DLLs live
        # This is needed because onnxruntime DLLs depend on VC++ runtime
        internal_dir = dist_dir / "_internal"
        ort_capi = internal_dir / "onnxruntime" / "capi"
        if ort_capi.exists():
            for dll_file in ort_capi.glob("*.dll"):
                target = internal_dir / dll_file.name
                if not target.exists():
                    print(f"   Copying {dll_file.name} to _internal/")
                    shutil.copy(dll_file, target)
            # Also copy the .pyd file
            for pyd_file in ort_capi.glob("*.pyd"):
                target = internal_dir / pyd_file.name
                if not target.exists():
                    print(f"   Copying {pyd_file.name} to _internal/")
                    shutil.copy(pyd_file, target)
    
    print(f"\nBuild successful!")
    print(f"   Output: {dist_dir}")
    print(f"   Executable: {exe_path}")
    print(f"   Size: {exe_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    # List contents
    print(f"\n   Contents:")
    total_size = 0
    for f in sorted(dist_dir.iterdir()):
        size = f.stat().st_size if f.is_file() else sum(p.stat().st_size for p in f.rglob("*") if p.is_file())
        total_size += size
        print(f"     {f.name}: {size / 1024 / 1024:.1f} MB")
    print(f"   Total: {total_size / 1024 / 1024:.1f} MB")
    
    return dist_dir


def create_zip(dist_dir: Path, version: str) -> Path:
    """Create a zip archive of the distribution."""
    import zipfile
    
    zip_name = f"marvin-onnx-{sys.platform}-v{version}"
    zip_path = dist_dir.parent / f"{zip_name}.zip"
    
    print(f"\nCreating archive: {zip_path}")
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for file in dist_dir.rglob("*"):
            if file.is_file():
                arcname = f"{zip_name}/{file.relative_to(dist_dir)}"
                zf.write(file, arcname)
    
    print(f"   Archive size: {zip_path.stat().st_size / 1024 / 1024:.1f} MB")
    return zip_path


def main():
    parser = argparse.ArgumentParser(description="Build Marvin ONNX executable")
    parser.add_argument(
        "--version", "-v",
        default="1.0.0",
        help="Version string for the archive name"
    )
    parser.add_argument(
        "--no-zip",
        action="store_true",
        help="Don't create a zip archive"
    )
    args = parser.parse_args()
    
    repo_root = Path(__file__).parent.parent
    
    print("=" * 60)
    print("Marvin ONNX UCI Engine - Executable Builder")
    print("=" * 60)
    
    # Check prerequisites
    if not check_pyinstaller():
        print("\nPyInstaller not installed!")
        print("   Install with: pip install pyinstaller")
        sys.exit(1)
    print("PyInstaller available")
    
    if not check_onnx_model(repo_root):
        sys.exit(1)
    
    # Build
    dist_dir = build_executable(repo_root)
    
    # Create archive
    if not args.no_zip:
        create_zip(dist_dir, args.version)
    
    print("\n" + "=" * 60)
    print("Done! Configure your chess GUI to use:")
    if sys.platform == "win32":
        print(f"  {dist_dir / 'marvin-onnx.exe'}")
    else:
        print(f"  {dist_dir / 'marvin-onnx'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
