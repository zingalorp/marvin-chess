#!/usr/bin/env python3
"""
Build release package for Marvin ONNX UCI Engine.

This script creates a distributable zip file containing:
- uci_onnx.py (the self-contained UCI engine)
- ONNX model files
- Launcher scripts for Windows/Linux
- Requirements and README

Usage:
    python scripts/build_release.py --version 1.0.0
"""

import argparse
import shutil
import zipfile
from pathlib import Path


def build_release(version: str, output_dir: Path | None = None) -> Path:
    """Build the release package."""
    
    repo_root = Path(__file__).parent.parent
    release_dir = repo_root / "release"
    output_dir = output_dir or repo_root / "dist"
    output_dir.mkdir(exist_ok=True)
    
    # Create staging directory
    staging_dir = output_dir / f"marvin-onnx-v{version}"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir()
    
    print(f"Building Marvin ONNX UCI Engine v{version}")
    print(f"  Staging: {staging_dir}")
    
    # Copy UCI engine
    uci_src = repo_root / "inference" / "uci_onnx.py"
    if not uci_src.exists():
        raise FileNotFoundError(f"UCI engine not found: {uci_src}")
    shutil.copy(uci_src, staging_dir / "uci_onnx.py")
    print(f"  Copied uci_onnx.py")
    
    # Copy ONNX model
    onnx_model = repo_root / "inference" / "marvin_small.onnx"
    onnx_data = repo_root / "inference" / "marvin_small.onnx.data"
    
    if not onnx_model.exists():
        print(f"  ONNX model not found. Run 'python scripts/export_onnx.py' first.")
        print(f"    Expected: {onnx_model}")
    else:
        shutil.copy(onnx_model, staging_dir / "marvin_small.onnx")
        print(f"  Copied marvin_small.onnx ({onnx_model.stat().st_size / 1024 / 1024:.1f} MB)")
    
    if not onnx_data.exists():
        print(f"  ONNX data file not found: {onnx_data}")
    else:
        shutil.copy(onnx_data, staging_dir / "marvin_small.onnx.data")
        print(f"  Copied marvin_small.onnx.data ({onnx_data.stat().st_size / 1024 / 1024:.1f} MB)")
    
    # Copy release files
    for filename in ["README.md", "requirements.txt", "run_engine.bat", "run_engine.sh"]:
        src = release_dir / filename
        if src.exists():
            shutil.copy(src, staging_dir / filename)
            print(f"  Copied {filename}")
        else:
            print(f"  Not found: {filename}")
    
    # Create version file
    (staging_dir / "VERSION").write_text(f"{version}\n")
    print(f"  Created VERSION file")
    
    # Create zip archive
    zip_path = output_dir / f"marvin-onnx-v{version}.zip"
    print(f"\n  Creating archive: {zip_path}")
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for file in staging_dir.rglob("*"):
            if file.is_file():
                arcname = file.relative_to(staging_dir)
                zf.write(file, arcname)
                print(f"    + {arcname}")
    
    zip_size_mb = zip_path.stat().st_size / 1024 / 1024
    print(f"\nRelease built: {zip_path} ({zip_size_mb:.1f} MB)")
    
    # Cleanup staging (optional - keep for inspection)
    # shutil.rmtree(staging_dir)
    
    return zip_path


def main():
    parser = argparse.ArgumentParser(description="Build Marvin ONNX release package")
    parser.add_argument(
        "--version", "-v",
        required=True,
        help="Version string (e.g., 1.0.0)"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output directory (default: dist/)"
    )
    args = parser.parse_args()
    
    build_release(args.version, args.output)


if __name__ == "__main__":
    main()
