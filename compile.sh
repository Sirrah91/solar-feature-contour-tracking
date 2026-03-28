#!/bin/bash

# --- Configuration ---
env="/clusterhome/david/.conda/envs/Contours/bin/python3.11"
CONDA_LIB_PATH="/clusterhome/david/.conda/envs/Contours/lib"
src_dir="/nfshome/david/Contours"
dist_dir="${src_dir}/python_compiled"
hooks_dir="${src_dir}/scr/devtools/hooks"

# Make sure the dist folder exists
mkdir -p "${dist_dir}"

# Install PyInstaller if not installed
"${env}" -m pip install pyinstaller

# --- List of Python scripts to compile ---
scripts=(\
    #"job_sizes.py" \
    #"run_image_to_stats.py" \
    #"run_split_to_phases.py" \
    #"run_contour_tracking.py" \
    #"run_sunspot_association.py" \
    #"run_sunspot_tracking_multilevel.py" \
    #"run_calc_stats.py" \
    )

cd "${src_dir}" || exit 1

for script in "${scripts[@]}"; do
    filename=$(basename -- "${script}")
    filename_no_ext="${filename%.*}"

    echo "=== Building ${filename_no_ext} ==="

    # Clean previous build artifacts
    rm -rf build dist "${dist_dir:?}/${filename_no_ext:?}" "${dist_dir:?}/${filename_no_ext:?}_bin"

    # --- Generate .spec file ---
    "${env}" -m PyInstaller.utils.cliutils.makespec \
        --noupx \
        --paths="${CONDA_LIB_PATH}" \
        --additional-hooks-dir="${hooks_dir}" \
        "${script}"

    specfile="./${filename_no_ext}.spec"
    if [[ ! -f "${specfile}" ]]; then
        echo "Error: Spec file not found for ${script}"
        exit 1
    fi

    # Inject recursion limit at the top of the spec
    sed -i '1i import sys; sys.setrecursionlimit(sys.getrecursionlimit() * 5)' "${specfile}"

    # --- Build with LD_LIBRARY_PATH set so PyInstaller finds the correct .so files ---
    LD_LIBRARY_PATH="${CONDA_LIB_PATH}:${LD_LIBRARY_PATH}" \
    "${env}" -m PyInstaller --clean --distpath="${dist_dir}" "${specfile}"

    # Move .spec to the compiled folder
    mkdir -p "${dist_dir}/${filename_no_ext}"
    mv "${specfile}" "${dist_dir}/${filename_no_ext}/${filename_no_ext}.spec"

    # Create a symlink for convenience
    ln -sf "${dist_dir}/${filename_no_ext}/${filename_no_ext}" \
           "${dist_dir}/${filename_no_ext}_bin"

    # Cleanup temporary build files
    rm -rf ./build

    echo "=== Done building ${filename_no_ext} ==="
done
