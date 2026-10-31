#!/bin/bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
env="/clusterhome/david/.conda/envs/Contours/bin/python3.11"
CONDA_LIB_PATH="/clusterhome/david/.conda/envs/Contours/lib"
src_dir="/nfshome/david/Contours"
dist_dir="${src_dir}/python_compiled"
hooks_dir="${src_dir}/src/devtools/hooks"
rthooks_dir="${src_dir}/src/devtools/rthooks"

# ---------------------------------------------------------------------------
# Scripts to compile.
# Paths are relative to src_dir, so subdirectory cli work fine.
# ---------------------------------------------------------------------------
scripts=(
    #"./src/cli/utils/job_sizes.py"
    #"./src/cli/run_image_to_stats.py"
    #"./src/cli/run_split_to_phases.py"
    #"./src/cli/run_contour_tracking.py"
    #"./src/cli/run_sunspot_association.py"
    #"./src/cli/run_sunspot_tracking_multilevel.py"
    #"./src/cli/run_calc_stats.py"
    #"./src/convective_regimes/cli/run_extract.py"
    #"./src/convective_regimes/cli/run_regression.py"
    #"./src/convective_regimes/cli/run_probabilities.py"
    #"./src/penumbra_evolution/cli/run_completeness.py"
    #"./src/penumbra_evolution/cli/run_background.py"
    "./src/penumbra_evolution/cli/run_companion.py"
    "./src/penumbra_evolution/cli/run_analysis.py"
)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
mkdir -p "${dist_dir}"

# Install PyInstaller only if not already present
if ! "${env}" -c "import PyInstaller" 2>/dev/null; then
    echo "=== Installing PyInstaller ==="
    "${env}" -m pip install pyinstaller
fi

cd "${src_dir}" || exit 1

# ---------------------------------------------------------------------------
# Build loop
# ---------------------------------------------------------------------------
for script in "${scripts[@]}"; do

    if [[ ! -f "${script}" ]]; then
        echo "Error: script not found: ${script}"
        exit 1
    fi

    filename=$(basename -- "${script}")
    name="${filename%.*}"

    echo ""
    echo "=== Building: ${name}  (${script}) ==="

    # Clean previous PyInstaller working directories
    rm -rf ./build ./dist "${dist_dir:?}/${name:?}" "${dist_dir:?}/${name:?}_bin"

    # Generate .spec file.
    # --name ensures the spec is always called <name>.spec in cwd,
    # regardless of how deep the script path is.
    "${env}" -m PyInstaller.utils.cliutils.makespec \
        --noupx \
        --name="${name}" \
        --paths="${CONDA_LIB_PATH}" \
        --additional-hooks-dir="${hooks_dir}" \
        --runtime-hook="${rthooks_dir}/rthook-environment.py" \
        "${script}"

    specfile="./${name}.spec"

    if [[ ! -f "${specfile}" ]]; then
        echo "Error: spec file not generated for ${script}"
        exit 1
    fi

    # Increase recursion limit at the top of the spec
    sed -i '1i import sys; sys.setrecursionlimit(sys.getrecursionlimit() * 5)' "${specfile}"

    # Build binary
    LD_LIBRARY_PATH="${CONDA_LIB_PATH}:${LD_LIBRARY_PATH:-}" \
    "${env}" -m PyInstaller \
        --clean \
        --distpath="${dist_dir}" \
        "${specfile}"

    # Move spec alongside the compiled output
    mkdir -p "${dist_dir}/${name}"
    mv "${specfile}" "${dist_dir}/${name}/${name}.spec"

    # Convenience symlink: dist_dir/<name>_bin -> dist_dir/<name>/<name>
    ln -sf "${dist_dir}/${name}/${name}" \
           "${dist_dir}/${name}_bin"

    # Remove PyInstaller's working directory
    rm -rf ./build ./dist

    echo "=== Done: ${name} -> ${dist_dir}/${name}_bin ==="

done

echo ""
echo "=== All builds complete ==="
