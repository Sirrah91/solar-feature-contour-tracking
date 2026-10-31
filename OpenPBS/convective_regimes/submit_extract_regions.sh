#!/bin/bash

base_dir="/nfshome/david/Contours/"
WD="${base_dir}/python_compiled/"
LD="${base_dir}/log/"
TD="${base_dir}/OpenPBS/convective_regimes/"

mkdir -p "${LD}"

SETTINGS_BASE=" \
--sunspot_type removed \
--levels 550.0 605.0 \
"

cd "${TD}" || exit

for filter_mode in "sunspots" "pores"; do
    for phase in "forming" "stable" "decaying"; do
        SETTINGS="${SETTINGS_BASE} --filter_mode ${filter_mode} --phase ${phase}"
        qsub -l mem=10gb,oldcpu=False -v WD="${WD}",LD="${LD}",SETTINGS="${SETTINGS}" extract_regions.pbs
        sleep 1
    done
done