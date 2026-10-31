import os
import sys
from PyInstaller.utils.hooks import collect_dynamic_libs, collect_submodules

# sys.prefix will point to /clusterhome/david/.conda/envs/Contours
conda_root = sys.prefix
conda_lib = os.path.join(conda_root, "lib")

# 1. Collect the main cairo submodules
hiddenimports = collect_submodules("cairo")

# 2. Force PyInstaller to grab the actual shared objects from Conda
# This prevents it from accidentally picking up /lib/x86_64-linux-gnu/libcairo.so
binaries = collect_dynamic_libs("cairo")

# 3. Manually add the critical dependencies that cause the "FT_Get_Transform" error
# We look for them specifically in your Conda lib folder
critical_libs = ["libfreetype", "libfontconfig", "libpng", "libz"]

for lib_name in critical_libs:
    lib_path = os.path.join(conda_lib, f"{lib_name}.so")
    if os.path.exists(lib_path):
        # Add to binaries: (source_path, destination_in_bundle)
        binaries.append((lib_path, "."))
