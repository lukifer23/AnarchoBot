#!/usr/bin/env python
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

# Clear Python cache before importing
try:
    subprocess.run([sys.executable, "-m", "py_compile", str(SRC / "anarchobot_mlx" / "train_mlx.py")],
                   check=True, capture_output=True)
except:
    pass

for path in (SRC, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from anarchobot_mlx.train_mlx import main

if __name__ == "__main__":
    main()
