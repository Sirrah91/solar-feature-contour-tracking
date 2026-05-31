#!/bin/bash

base_dir="/nfshome/david/Contours/"
WD="${base_dir}/python_compiled/"
LD="${base_dir}/log/"
TD="${base_dir}/OpenPBS/convective_regimes/"

mkdir -p "${LD}"

SETTINGS=" \
--filter_modes sunspots pores \
--phases all forming stable decaying \
--quantities Bhor \
--boundary b605.0 \
--n_B_bins 150 \
--n_g_bins 45 \
--min_count 50 \
"
cd "${TD}" || exit

qsub -l mem=10gb,oldcpu=False -v WD="${WD}",LD="${LD}",SETTINGS="${SETTINGS}" probabilities.pbs
