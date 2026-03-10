from PyInstaller.utils.hooks import collect_submodules, collect_data_files

# Include all Python modules and extension modules
hiddenimports = collect_submodules("skimage")

# Include any non-code data files from the package
datas = collect_data_files("skimage", includes=["*.pyi", "**/*.pyi"])
