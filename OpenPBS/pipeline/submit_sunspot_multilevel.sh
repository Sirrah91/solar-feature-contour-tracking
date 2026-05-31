#!/bin/bash

base_dir="/nfshome/david/Contours/"
WD="${base_dir}/python_compiled/"
LD="${base_dir}/log/"
TD="${base_dir}/OpenPBS/pipeline/"

mkdir -p "${LD}"

# "--data_dir DIR": Directory containing input FITS files.

# "--contour_quantity QUANTITY": Quantity used to compute the contour.
# "--component LEVEL [MIN_AREA] [REGION]": Define a component to associate with the tracked outer contours.
#                                          You can repeat --component multiple times to define multiple components.

# "--max_gap INT": Maximum number of consecutive frames a tracked contour can be absent
#                  and still be considered part of a valid track (default 3).
# "--min_frames INT": Minimum number of image frames in which the contour must appear (default 0).
# "--collapse_nested": Collapse nested contours (flag, default False).
#                      Cannot be set together with remove_nested.
# "--remove_nested": Remove nested frames from outer tracks (flag, default False).
#                    Cannot be set together with collapse_nested.
# "--min_vertices INT": Minimum number of vertices a contour must have to be kept (default 4).
# "--max_healing_gap FLOAT": Maximum distance (in pixels) between first and last contour point
#                            to automatically 'snap' them together if the contour is almost closed (default 0.0).
# "--max_closing_gap FLOAT": Maximum distance (in pixels) between first and last contour point
#                            to forcibly close the contour by appending the first point (default 0.0).

# "--iou_threshold FLOAT": Minimum IoU (Intersection over Union) required to merge
#                          two contours into one region (default 0.3).
# "--registration": Enable image alignment (registration) before contour tracking (flag, default False).

# "--containment_mode MODE": Containment criterion for component association (default covers).
# "--min_containment FLOAT": Minimum containment fraction (robust mode only, default 0.8).

# "--sunspot_outdir DIR": Output directory for saving results (default /nfsscratch/david/Contours/sunspots).
# "--sunspot_output_name BASENAME": Base name for saved output files. If empty, it is constructed automatically (default empty).


SETTINGS_NO_DATADIR=" \
--contour_quantity Ic \
--component 0.9 5.0 \
--component 0.65 3.0 \
--component 0.5 3.0 \
--max_gap 3 \
--min_frames 3 \
--collapse_nested \
--iou_threshold 0.3 \
--registration \
--containment_mode covers \
--min_containment 0.8 \
--sunspot_outdir /nfsscratch/david/Contours/sunspots_multilevel
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
    qsub -l mem="${SIZE[0]}"gb,oldcpu=False -v WD="${WD}",LD="${LD}",SETTINGS="${SETTINGS}" sunspot_multilevel.pbs
    sleep 1
done

echo "All done"
