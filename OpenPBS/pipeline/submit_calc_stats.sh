#!/bin/bash

base_dir="/nfshome/david/Contours/"
WD="${base_dir}/python_compiled/"
LD="${base_dir}/log/"
TD="${base_dir}/OpenPBS/pipeline/"

mkdir -p "${LD}"

# "--sunspot_input_path PATH": Input file with sunspots (.npz).

# "--quantities QUANTITIES, ...": List of quantities for which the statistics are computed (Ic, B, Bp, Bt, Br, or Bhor).

# "--header_index INT": Header index to use for the foreshortening correction (default 0).
# "--max_vertex_spacing FLOAT": Minimum step between vertices in px (default 0.5).

# "--stats_outdir DIR": Output directory for saving results (default /nfsscratch/david/Contours/sunspots_stats).
# "--stats_output_name BASENAME": Base name for saved output files. If empty, it is constructed automatically (default empty).


SETTINGS_NO_SUNSPOTFILE=" \
--quantities Ic B Bp Bt Br Bhor \
--header_index 0 \
--max_vertex_spacing 0.5 \
--stats_outdir /nfsscratch/david/Contours/sunspots_collapsed_stats \
"

# run from largest file
DIR="/nfsscratch/david/Contours/sunspots_collapsed"

find "${DIR}" -maxdepth 1 -type f -name '*.npz' -printf '%s\t%p\n' \
  | sort -nr \
  | cut -f2- \
  | while IFS= read -r file; do

      SETTINGS="${SETTINGS_NO_SUNSPOTFILE} --sunspot_input_path ${file}"

      cd "${WD}" || exit
      SIZE=(10)

      echo "Sunspot calc-statistics job size: ${SIZE[0]} GB"
      cd "${TD}" || exit

      # Submit the job
      qsub -l mem="${SIZE[0]}"gb,oldcpu=False -v WD="${WD}",LD="${LD}",SETTINGS="${SETTINGS}" calc_stats.pbs
      sleep 1
    done

echo "All done"
