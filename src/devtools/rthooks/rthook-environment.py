"""Runtime environment setup for PyInstaller applications."""

import os

SYSTEM_BIN = (
    "/bin",
    "/usr/bin",
    "/usr/bin/latex",
)

current_path = os.environ.get("PATH", "")
current_paths = current_path.split(os.pathsep) if current_path else []

missing_paths = [p for p in SYSTEM_BIN if p not in current_paths]
if missing_paths:
    os.environ["PATH"] = os.pathsep.join([*missing_paths, *current_paths])
