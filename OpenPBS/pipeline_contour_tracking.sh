#!/bin/bash

base_dir="/nfshome/david/Contours/"
WD="${base_dir}/python_compiled/"
LD="${base_dir}/log/"
TD="${base_dir}/OpenPBS/"

mkdir -p "${LD}"

# "--data_dir DIR": Directory containing input FITS files.

# "--contour_quantity QUANTITY": Quantity used to compute the contour.
# "--contour_level LEVEL": Threshold value to define the contour.

# "--min_area FLOAT": Minimum contour area in pixels² for contours to be considered (default 3.0).
# "--max_gap INT": Maximum number of consecutive frames a tracked contour can be absent
#                  and still be considered part of a valid track (default 3).

# "--iou_threshold FLOAT": Minimum IoU (Intersection over Union) required to merge
#                          two contours into one region (default 0.3).
# "--registration": Enable image alignment (registration) before contour tracking (flag, default False).

# "--track_outdir DIR": Output directory for saving results (default /nfsscratch/david/Contours/tracks).
# "--track_output_name BASENAME": Base name for saved output files. If empty, it is constructed automatically (default empty).


SETTINGS_NO_DATADIR=" \
--contour_quantity Ic \
--contour_level 0.9 \
--min_area 3.0 \
--max_gap 3 \
--iou_threshold 0.3 \
--registration \
"

# Loop over each subfolder in input_base
for data_dir_full in "/nfsscratch/david/NN/results/"*/ ; do
    # Remove trailing slash for cleaner naming logic in Python
    data_dir="${data_dir_full%/}"

    SETTINGS="${SETTINGS_NO_DATADIR} --data_dir ${data_dir}"

    cd "${WD}" || exit
    SIZE=($(./job_sizes_bin --data_dir "${data_dir}" --parse_quantities))

    echo "Contour tracking job size: ${SIZE[0]} GB"
    cd "${TD}" || exit

    # Submit the job
    qsub -l mem="${SIZE[0]}"gb,oldcpu=False -v WD="${WD}",LD="${LD}",SETTINGS="${SETTINGS}" contour_tracking.pbs
    sleep 1
done

echo "All done"
