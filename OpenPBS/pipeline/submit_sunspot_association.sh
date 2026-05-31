#!/bin/bash

base_dir="/nfshome/david/Contours/"
WD="${base_dir}/python_compiled/"
LD="${base_dir}/log/"
TD="${base_dir}/OpenPBS/pipeline/"

mkdir -p "${LD}"

# "--track_input_path PATH": Input file with tracked outer contours (.npz).

# "--inner_contour_quantity QUANTITY": Quantity used to compute the contour.
# "--component LEVEL [MIN_AREA] [REGION]": Define a component to associate with the tracked outer contours.
#                                          You can repeat --component multiple times to define multiple components.

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

# "--containment_mode MODE": Containment criterion for component association (default covers).
# "--min_containment FLOAT": Minimum containment fraction (robust mode only, default 0.8).

# "--sunspot_outdir DIR": Output directory for saving results (default /nfsscratch/david/Contours/sunspots).
# "--sunspot_output_name BASENAME": Base name for saved output files. If empty, it is constructed automatically (default empty).


SETTINGS_NO_TRACKFILE=" \
--inner_contour_quantity Ic \
--component 0.65 3.0 \
--component 0.5 3.0 \
--min_frames 3 \
--remove_nested \
--containment_mode covers \
--min_containment 0.8 \
--sunspot_outdir /nfsscratch/david/Contours/sunspots_removed \
"

# Loop over the track files
for track_path in "/nfsscratch/david/Contours/tracks/"*.npz ; do

    SETTINGS="${SETTINGS_NO_TRACKFILE} --track_input_path ${track_path}"

    cd "${WD}" || exit
    SIZE=(3)

    echo "Sunspot association job size: ${SIZE[0]} GB"
    cd "${TD}" || exit

    # Submit the job
    qsub -l mem="${SIZE[0]}"gb,oldcpu=False -v WD="${WD}",LD="${LD}",SETTINGS="${SETTINGS}" sunspot_association.pbs
    sleep 1
done

echo "All done"
