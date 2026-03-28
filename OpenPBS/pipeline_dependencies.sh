#!/bin/bash

base_dir="/nfshome/david/Contours/"
WD="${base_dir}/python_compiled/"
LD="${base_dir}/log/"
TD="${base_dir}/OpenPBS/"

mkdir -p "${LD}"

CONTOUR_SETTINGS_NO_DATADIR=" \
--contour_quantity Ic \
--contour_level 0.9 \
--min_area 3.0 \
--max_gap 3 \
--iou_threshold 0.3 \
--registration \
"

ASSOC_SETTINGS_NO_TRACKFILE=" \
--inner_contour_quantity Ic \
--component 0.65 3.0 \
--component 0.5 3.0 \
--min_frames 3 \
--collapse_nested \
--containment_mode covers \
--min_containment 0.8 \
"

STATS_SETTINGS_NO_FILE=" \
--quantities Ic B Bp Bt Br Bhor \
--header_index 0 \
--max_vertex_spacing 0.5 \
"

SPLIT_SETTINGS=" \
--stats_dir /nfsscratch/david/Contours/sunspots_stats \
"

# Loop over each subfolder in input_base
for data_dir_full in "/nfsscratch/david/NN/results/"*/ ; do
    # Remove trailing slash for cleaner naming logic in Python
    data_dir="${data_dir_full%/}"

    SETTINGS=" \
    ${CONTOUR_SETTINGS_NO_DATADIR} --data_dir ${data_dir} \
    ${ASSOC_SETTINGS_NO_TRACKFILE} \
    ${STATS_SETTINGS_NO_FILE} \
    "

    cd "${WD}" || exit
    SIZE=(10)

    echo "Master pipeline job size: ${SIZE[0]} GB"
    cd "${TD}" || exit

    # Submit the job
    ID=$(qsub -l mem="${SIZE[0]}"gb,oldcpu=False -v WD="${WD}",LD="${LD}",SETTINGS="${SETTINGS}" image_to_stats.pbs)

    # Append ID to a colon-separated list
    JOB_IDS="${JOB_IDS}${JOB_IDS:+:}${ID}"

    sleep 1
done

# Submit the final job
qsub -W depend=afterok:"${JOB_IDS}" -l mem=15gb,oldcpu=False -v WD="${WD}",LD="${LD}",SETTINGS="${SPLIT_SETTINGS}" split_to_phases.pbs

echo "All done"
