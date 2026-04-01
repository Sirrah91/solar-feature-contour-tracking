from os import path
from tqdm import tqdm
import numpy as np
from scipy.stats import gaussian_kde
import plotly.graph_objects as go

from scr.config.paths import PATH_SUNSPOTS_PHASES
from scr.config.filtering import gimme_filtering_kwargs
from scr.io.parquet import load_parquet
from scr.io.npz import load_npz, save_npz

from scr.io.fits.read import load_image
from scr.statistics.dataframe.filtering import filter_combined_df
from scr.geometry.contours.extraction import find_contours
from scr.geometry.raster.binary import contours_to_binary_mask
from scr.contours.selection import select_support_contours
from scr.physics.magnetic import compute_unsigned_inclination
from scr.plotting.generic.volume import plot_volume_3d
from scr.devtools.parser import CustomArgumentParser, CustomFormatter, inplace_process_nargs1_args
'''
parser = CustomArgumentParser(
    allow_abbrev=False,
    add_help=False,
    description="Calculate memory requirements to process sunspot contours.\n\n"
                "Example:\n"
                "  python job_sizes.py --data_dir /path/to/data",
    formatter_class=CustomFormatter
)
# Core options
core = parser.add_argument_group("core settings")
core.add_argument(
    "--filter_mode",
    type=str,
    required=True,
    nargs=1,
    help="Directory containing input files."
)
# Core options
core = parser.add_argument_group("core settings")
core.add_argument(
    "--phase",
    type=str,
    required=True,
    nargs=1,
    help="Directory containing input files."
)

# To be able to pass all ${SETTINGS}
args, _ = parser.parse_known_args()

# convert list[arg] to arg
inplace_process_nargs1_args(
    parser=parser,
    args=args
)

filter_mode = args.filter_mode
phase = args.phase

df = load_parquet(
    path.join(PATH_SUNSPOTS_PHASES, f"all_sunspots_phases_merged.parquet")
)
for filter_mode in ["sunspots", "pores"]:
    for phase in ["forming", "stable", "decaying"]:
        df_filtered = filter_combined_df(df, gimme_filtering_kwargs(filter_mode) | {"phase": {"mode": "frame-wise", "exact_value": phase}})

        ic_ic05, b_ic05, bhor_ic05, binc_ic05 = [], [], [], []
        ic_ic09, b_ic09, bhor_ic09, binc_ic09 = [], [], [], []
        ic_b550, b_b550, bhor_b550, binc_b550 = [], [], [], []

        obs_id = df_filtered.groupby(["observation_id"], observed=True)

        for npz_filename, group in tqdm(obs_id, desc="OBS ID"):
            sunspots = load_npz(npz_filename[0])["tracks"].item()["sunspots"]
            for image_filename, frame in zip(group["image_path"], group["frame"]):
                ic_image = load_image(image_filename, quantity="Ic")
                b_image = load_image(image_filename, quantity="B")
                b_hor_image = load_image(image_filename, quantity="Bhor")
                b_inc_image = load_image(image_filename, quantity="Binc")
                b_inc_image = compute_unsigned_inclination(signed_inclination=b_inc_image)

                outer_contours = []
                inner_contours = []
                for sunspot_id, sunspot in sunspots.items():
                    if frame in sunspot["outer"] and sunspot["outer"][frame]:
                        outer_contours += sunspot["outer"][frame]
                    if frame in sunspot["inner"] and sunspot["inner"][frame]:
                        inner_contours += sunspot["inner"][frame]

                b_contours = find_contours(b_image, level=550.0)  # outer
                b_contours = select_support_contours(outer_contours, b_contours)

                umbra_mask = contours_to_binary_mask(inner_contours, shape=b_image.shape)
                penumbra_mask = contours_to_binary_mask(outer_contours, shape=b_image.shape)
                b_mask = contours_to_binary_mask(b_contours, shape=b_image.shape)

                b_mask = b_mask & ~ penumbra_mask
                penumbra_mask = penumbra_mask & ~umbra_mask

                ic_b550.append(ic_image[b_mask])
                b_b550.append(b_image[b_mask])
                bhor_b550.append(b_hor_image[b_mask])
                binc_b550.append(b_inc_image[b_mask])

                ic_ic09.append(ic_image[penumbra_mask])
                b_ic09.append(b_image[penumbra_mask])
                bhor_ic09.append(b_hor_image[penumbra_mask])
                binc_ic09.append(b_inc_image[penumbra_mask])

                ic_ic05.append(ic_image[umbra_mask])
                b_ic05.append(b_image[umbra_mask])
                bhor_ic05.append(b_hor_image[umbra_mask])
                binc_ic05.append(b_inc_image[umbra_mask])

        save_npz(filename=f"/nfshome/david/Contours/tests/{filter_mode}_{phase}.npz",
                 ic_b550=np.asarray(ic_b550, dtype=object),
                 b_b550=np.asarray(b_b550, dtype=object),
                 bhor_b550=np.asarray(bhor_b550, dtype=object),
                 binc_b550=np.asarray(binc_b550, dtype=object),
                 ic_ic09=np.asarray(ic_ic09, dtype=object),
                 b_ic09=np.asarray(b_ic09, dtype=object),
                 bhor_ic09=np.asarray(bhor_ic09, dtype=object),
                 binc_ic09=np.asarray(binc_ic09, dtype=object),
                 ic_ic05=np.asarray(ic_ic05, dtype=object),
                 b_ic05=np.asarray(b_ic05, dtype=object),
                 bhor_ic05=np.asarray(bhor_ic05, dtype=object),
                 binc_ic05=np.asarray(binc_ic05, dtype=object),
                 phase=phase,
                 filter_mode=filter_mode)
'''

for filter_mode in ["sunspots", "pores"]:
    for phase in ["forming", "stable", "decaying"]:
        data = load_npz(f"/nfshome/david/Contours/tests/{filter_mode}_{phase}.npz")

        # Ic = np.concatenate(data["ic_ic05"])
        # B = np.concatenate(data["b_ic05"])
        # Bhor = np.concatenate(data["bhor_ic05"])

        # umbra = np.vstack([Ic, B, Bhor])

        Ic = np.concatenate(list(data["ic_b550"]) + list(data["ic_ic09"]) + list(data["ic_ic05"]))
        B = np.concatenate(list(data["b_b550"]) + list(data["b_ic09"]) + list(data["b_ic05"]))
        Bhor = np.concatenate(list(data["bhor_b550"]) + list(data["bhor_ic09"]) + list(data["bhor_ic05"]))
        Binc = np.rad2deg(np.concatenate(list(data["binc_b550"]) + list(data["binc_ic09"]) + list(data["binc_ic05"])))

        IBBh = np.vstack([Ic, B, Bhor])
        IBiBh = np.vstack([Ic, Binc, Bhor])
        del Ic, B, Bhor, Binc

        IBBh = IBBh[:, np.all(np.isfinite(IBBh), axis=0)]
        IBiBh = IBiBh[:, np.all(np.isfinite(IBiBh), axis=0)]

        xmin, xmax = 0.1, 1.3
        ymin, ymax = 0, 3000
        zmin, zmax = 0, 2500
        inc_min, inc_max = 0, 90

        N = 1000000

        X, Y, Z = np.mgrid[
            xmin:xmax:complex(0, 25),
            ymin:ymax:complex(0, 61),
            zmin:zmax:complex(0, 51)
        ]
        X2, Y2, Z2 = np.mgrid[
            xmin:xmax:complex(0, 25),
            inc_min:inc_max:complex(0, 61),
            zmin:zmax:complex(0, 51)
        ]

        positions = np.vstack([X.ravel(), Y.ravel(), Z.ravel()])
        positions2 = np.vstack([X2.ravel(), Y2.ravel(), Z2.ravel()])

        inds = np.random.choice(np.shape(IBBh)[1], size=min(N, np.shape(IBBh)[1]), replace=False)
        kde_IBBh = gaussian_kde(IBBh[:, inds])
        inds = np.random.choice(np.shape(IBiBh)[1], size=min(N, np.shape(IBiBh)[1]), replace=False)
        kde_IBiBh = gaussian_kde(IBiBh[:, inds])
        del IBBh, IBiBh
        print("loaded")

        density_IBBh = kde_IBBh(positions).reshape(X.shape)
        density_IBiBh = kde_IBiBh(positions2).reshape(X.shape)
        print("kde done")

        density_IBBh_log = np.log(density_IBBh)
        density_IBiBh_log = np.log(density_IBiBh)

        # Plot
        print("rendering")
        fig = go.Figure()
        plot_volume_3d(
            fig, X, Y, Z, density_IBBh_log,
            name="B > 550 G",
            isomin=np.nanpercentile(density_IBBh_log, 80),
        )

        fig.update_layout(
            scene=dict(
                xaxis=dict(title="Ic"),
                yaxis=dict(title="B (G)"),
                zaxis=dict(title="Bhor (G)"),
            )
        )

        fig.write_html(f"/nfshome/david/Contours/tests/{filter_mode}_{phase}_I-B-Bhor.html")

        # Plot
        print("rendering")
        fig = go.Figure()
        plot_volume_3d(
            fig, X2, Y2, Z2, density_IBiBh_log,
            name="B > 550 G",
            isomin=np.nanpercentile(density_IBiBh_log, 80),
        )

        fig.update_layout(
            scene=dict(
                xaxis=dict(title="Ic"),
                yaxis=dict(title="B inc (deg)"),
                zaxis=dict(title="Bhor (G)"),
            )
        )

        fig.write_html(f"/nfshome/david/Contours/tests/{filter_mode}_{phase}_I-Binc-Bhor.html")
