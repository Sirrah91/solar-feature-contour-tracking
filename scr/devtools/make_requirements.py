"""
# Create requirements.txt and pyproject.toml by following these steps:
#
# 1) Ensure uv is installed:
#     pip install uv pigar
#
# 2) Run this script from the project root or via:
#     python ./scr/devtools/make_requirements.py
#
# This script automates the following:
#   - Scans code for imports using 'pigar'.
#   - Generates 'pyproject.toml'.
#   - Uses 'uv' to compile the final, pinned 'requirements.txt'.
"""

import subprocess
import os
import sys
import shutil
import importlib.util

# --- BOOTSTRAP PATHS ---
script_abs_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_abs_path)))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from scr.config.paths import PROJECT_DIR
except ImportError:
    PROJECT_DIR = project_root

# --- CONFIGURATION ---
STRICT_IGNORE_PATHS = ["graphic_output", "log", "OpenPBS", "python_compiled", "tests", "venv", ".git"]

NAME_MAP = {
    "skimage": "scikit-image",
    "cv2": "opencv-python",
    "PIL": "Pillow",
    "yaml": "PyYAML",
    "mpl_toolkits": "matplotlib",
    "cairo": "pycairo",
}

FORCED_DEPS = [
    "pycairo", "scikit-image", "pyarrow", "PyInstaller"
]

IGNORE_PKGS = []


def clean_precompiled():
    print("--> Cleaning __pycache__...")
    for root, dirs, files in os.walk(PROJECT_DIR):
        if any(x in root for x in STRICT_IGNORE_PATHS):
            continue
        for d in dirs:
            if d == "__pycache__":
                shutil.rmtree(os.path.join(root, d), ignore_errors=True)


def run_step_1_scan():
    print("\n--- Step 1: Scanning project (pigar) ---")
    ignore_str = f"-i {','.join(STRICT_IGNORE_PATHS)}"
    # Local scan only to avoid InvalidUrl errors
    cmd = f"yes | pigar generate --auto-select -f requirements.tmp {ignore_str} {PROJECT_DIR}"
    subprocess.run(cmd, shell=True, capture_output=True, text=True)


def run_step_2_create_toml():
    print("\n--- Step 2: Generating pyproject.toml ---")
    deps = set(FORCED_DEPS)
    tmp_path = os.path.join(PROJECT_DIR, "requirements.tmp")

    if os.path.exists(tmp_path):
        with open(tmp_path, "r") as f:
            for line in f:
                if "referenced from" in line or not line.strip() or line.startswith("#"):
                    continue
                name = line.split("==")[0].strip()
                clean_name = NAME_MAP.get(name, name)
                if clean_name and clean_name not in IGNORE_PKGS:
                    deps.add(clean_name)

    toml_content = f"""[project]
name = "{os.path.basename(PROJECT_DIR)}"
version = "2.0.0"
description = "Auto-generated from nested source scan"
authors = [
    {{ name = "David Korda", email = "david.korda@asu.cas.cz" }}
]
license = "MIT"
requires-python = ">=3.9, <3.14"
dependencies = [
"""
    for dep in sorted(deps):
        toml_content += f'    "{dep}",\n'
    toml_content += """]

[build-system]
requires = ["setuptools", "wheel"]
build-backend = "setuptools.build_meta"
"""
    with open(os.path.join(PROJECT_DIR, "pyproject.toml"), "w") as f:
        f.write(toml_content)


def run_step_3_uv_compile():
    print("\n--- Step 3: Compiling with uv ---")

    # uv command structure
    # --no-cache ensures it fetches the latest if needed
    cmd = ["uv", "pip", "compile", "pyproject.toml", "-o", "requirements.txt"]

    print(f"--> Executing: {' '.join(cmd)}")
    try:
        # shell=False is safer here as uv is a binary
        subprocess.check_call(cmd)
        print("--> requirements.txt successfully generated via uv.")
    except Exception as e:
        print(f"\n[!] uv compilation failed. Check if uv is installed: 'pip install uv'")
        print(f"Details: {e}")


def main():
    os.chdir(PROJECT_DIR)

    # Check for uv binary in PATH
    if shutil.which("uv") is None:
        print("Error: 'uv' not found in PATH. Please run: pip install uv")
        sys.exit(1)

    if importlib.util.find_spec("pigar") is None:
        print("Error: 'pigar' not found. Please run: pip install pigar")
        sys.exit(1)

    clean_precompiled()
    run_step_1_scan()
    run_step_2_create_toml()
    run_step_3_uv_compile()

    tmp_path = os.path.join(PROJECT_DIR, "requirements.tmp")
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    print("\nWorkflow complete! Final files: pyproject.toml, requirements.txt")


if __name__ == "__main__":
    main()
