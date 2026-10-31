#!/bin/bash
env="/clusterhome/david/.conda/envs/Contours/bin/python3.11"
src_dir="/nfshome/david/Contours"

SETTINGS=" \
--plots all \
--phases all forming stable decaying \
--fig_format pdf \
"

cd "${src_dir}" || exit 1

"${env}" -m scr.convective_regimes.scripts.run_plot ${SETTINGS}
