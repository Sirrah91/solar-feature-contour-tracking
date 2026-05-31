#!/bin/bash

base_dir="/nfshome/david/Contours/"
WD="${base_dir}/python_compiled/"
LD="${base_dir}/log/"
TD="${base_dir}/OpenPBS/convective_regimes/"

mkdir -p "${LD}"

SETTINGS=" \
--boundary b550.0 \
--window_half_width 5.0 \
--min_samples 1000 \
--gamma_min 5.0 \
--gamma_max 86.0 \
--gamma_step 2.0 \
"
cd "${TD}" || exit

for filter_mode in "sunspots" "pores"; do
    for phase in "all" "forming" "stable" "decaying" ; do
        SETTINGS="--filter_modes ${filter_mode} --phases ${phase}"
        qsub -l mem=10gb,oldcpu=False -v WD="${WD}",LD="${LD}",SETTINGS="${SETTINGS}" regression.pbs
        sleep 1
    done
done