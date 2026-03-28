#!/bin/bash

base_dir="/nfshome/david/Contours/"
WD="${base_dir}/python_compiled/"
LD="${base_dir}/log/"
TD="${base_dir}/OpenPBS/"

mkdir -p "${LD}"

# "--stats_dir DIR": Directory containing input sunspot-stat files.

# "--slope_path PATH": File path for precomputed slopes. Can be absolute, or relative to --phases_outdir (default flux_slopes.parquet).
# "--collect_new_slopes": Flag to trigger new slope computation (flag, default False).

# "--phases_outdir DIR": Output directory for saving results (default /nfsscratch/david/Contours/sunspots_phases).
# "--phases_output_name BASENAME": Base name for saved output files. If empty, it is constructed automatically (default sunspots_phases).


SETTINGS=" \
--stats_dir /nfsscratch/david/Contours/sunspots_stats \
"

cd "${WD}" || exit
SIZE=(50)

echo "Sunspot split-to-phases job size: ${SIZE[0]} GB"
cd "${TD}" || exit

# Submit the job
qsub -l mem="${SIZE[0]}"gb,oldcpu=False -v WD="${WD}",LD="${LD}",SETTINGS="${SETTINGS}" split_to_phases.pbs
sleep 1

echo "All done"
