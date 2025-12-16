#!/usr/bin/env python
"""
Cache clearing script for AnarchoBot training.
Ensures latest code is loaded and no stale cache issues.
"""
import subprocess
import sys
from pathlib import Path

def clear_python_cache():
    """Clear Python bytecode cache"""
    root = Path(__file__).resolve().parents[1]

    # Remove __pycache__ directories
    for pycache in root.rglob("__pycache__"):
        if pycache.is_dir():
            try:
                import shutil
                shutil.rmtree(pycache)
                print(f"Removed: {pycache}")
            except Exception as e:
                print(f"Failed to remove {pycache}: {e}")

    # Remove .pyc files
    for pyc_file in root.rglob("*.pyc"):
        try:
            pyc_file.unlink()
            print(f"Removed: {pyc_file}")
        except Exception as e:
            print(f"Failed to remove {pyc_file}: {e}")

    # Recompile Python files
    try:
        subprocess.run([sys.executable, "-m", "compileall", str(root / "src")],
                      check=True, capture_output=True)
        print("Recompiled source files")
    except Exception as e:
        print(f"Failed to recompile: {e}")

if __name__ == "__main__":
    print("Clearing Python cache...")
    clear_python_cache()
    print("Cache clearing complete.")
