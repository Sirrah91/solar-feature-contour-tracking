import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from os import path

from scr.devtools.plotting import plot_me
from scr.config.paths import PATH_SUNSPOTS_PHASES
from scr.config.figures import CBAR_KWARGS
from scr.config.filtering import gimme_filtering_kwargs
from scr.io.parquet import load_parquet
from scr.statistics.dataframe.filtering import filter_combined_df
from scr.plotting.generic.hist import plot_hist2d
from scr.plotting.style.colorbar import add_colorbar
from scr.plotting.style.fonts import font_style

df = load_parquet(
    path.join(PATH_SUNSPOTS_PHASES, f"all_sunspots_phases_merged.parquet")
)
used_keys = [
    "phase",
    "penumbra_corrected_area",
    "Bhor_umbra_corrected_border_flux_mean",
    "Bhor_pore_corrected_border_flux_mean",
    "Bhor_penumbra_corrected_border_flux_mean",
    "B_umbra_corrected_border_flux_mean",
    "B_pore_corrected_border_flux_mean",
    "B_penumbra_corrected_border_flux_mean",
    "Ic_umbra_corrected_border_flux_mean",
    "Ic_pore_corrected_border_flux_mean",
    "Ic_penumbra_corrected_border_flux_mean",
]

df_spot = filter_combined_df(df, gimme_filtering_kwargs("sunspots"))
df_pore = filter_combined_df(df, gimme_filtering_kwargs("pores"))

# DO NOT SLICE BEFORE FILTERING...
df = df[used_keys]
df_spot = df_spot[used_keys]
df_pore = df_pore[used_keys]

phases = ("forming", "stable", "decaying")
regions = ("umbra", "pore", "penumbra")
figsize = (4 * len(phases), 3 * len(regions))

with font_style(fontsize=12):
    for df_base, obs_type in ((df_spot, "spot"), (df_pore, "pore"), (df, "all")):
        fig, axes = plt.subplots(
            nrows=len(regions),
            ncols=len(phases),
            squeeze=False,
            figsize=figsize,
            sharex=True,
            sharey=True,
        )
        for iphase, phase in enumerate(phases):
            df_phase = filter_combined_df(df_base, {"phase": {"exact_value": phase, "mode": "frame-wise"}})

            for iregion, region in enumerate(regions):
                ax = axes[iregion, iphase]
                im = plot_hist2d(
                    ax,
                    x=df_phase[f"Bhor_{region}_corrected_border_flux_mean"],
                    y=df_phase["penumbra_corrected_area"],
                    range=((300, 1500), (0, 30000)) if obs_type != "pore" else ((300, 1500), (0, 1000)),
                )

                if iphase == 0:
                    ax.set_ylabel(region)
                if iregion == 0:
                    ax.set_xlabel(phase)
                    ax.xaxis.set_label_position("top")

                """
                add_colorbar(
                    ax,
                    im,
                    cbar_kwargs=CBAR_KWARGS,
                    label=r"PDF (\%)",
                    formatter=FuncFormatter(lambda v, _: f"{100 * v:.3f}"),
                )
                """
        plt.tight_layout()

    plt.show()

#############################################################################################
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from os import path

from scr.devtools.plotting import plot_me
from scr.config.paths import PATH_SUNSPOTS_PHASES
from scr.config.figures import CBAR_KWARGS
from scr.config.filtering import gimme_filtering_kwargs
from scr.io.parquet import load_parquet
from scr.statistics.dataframe.filtering import filter_combined_df
from scr.plotting.generic.hist import plot_hist2d
from scr.plotting.style.colorbar import add_colorbar
from scr.plotting.style.fonts import font_style

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from skimage.measure import marching_cubes

df = load_parquet(
    path.join(PATH_SUNSPOTS_PHASES, f"all_sunspots_phases_merged.parquet")
)
used_keys = [
    "phase",
    "penumbra_corrected_area",
    "Bhor_umbra_corrected_border_flux_mean",
    "Bhor_pore_corrected_border_flux_mean",
    "Bhor_penumbra_corrected_border_flux_mean",
    "B_umbra_corrected_border_flux_mean",
    "B_pore_corrected_border_flux_mean",
    "B_penumbra_corrected_border_flux_mean",
    "Ic_umbra_corrected_border_flux_mean",
    "Ic_pore_corrected_border_flux_mean",
    "Ic_penumbra_corrected_border_flux_mean",
]

df_spot = filter_combined_df(df, gimme_filtering_kwargs("sunspots"))
df_pore = filter_combined_df(df, gimme_filtering_kwargs("pores"))

# DO NOT SLICE BEFORE FILTERING...
df = df[used_keys]
df_spot = df_spot[used_keys]
df_pore = df_pore[used_keys]

phases = ("forming", "stable", "decaying")
regions = ("umbra", "pore", "penumbra")
figsize = (4 * len(phases), 3 * len(regions))

df_spot_filtered = filter_combined_df(df_spot, {"B_penumbra_corrected_border_flux_mean": {"func": lambda s: s > 600, "mode": "frame-wise"}})
df_pore_filtered = filter_combined_df(df_pore, {"B_penumbra_corrected_border_flux_mean": {"func": lambda s: s > 600, "mode": "frame-wise"}})

df_spot_filtered_fo = filter_combined_df(df_spot, {"phase": {"func": lambda s: s == "forming", "mode": "frame-wise"}})
df_spot_filtered_st = filter_combined_df(df_spot, {"phase": {"func": lambda s: s == "stable", "mode": "frame-wise"}})
df_spot_filtered_de = filter_combined_df(df_spot, {"phase": {"func": lambda s: s == "decaying", "mode": "frame-wise"}})

df_pore_filtered_fo = filter_combined_df(df_pore, {"phase": {"func": lambda s: s == "forming", "mode": "frame-wise"}})
df_pore_filtered_st = filter_combined_df(df_pore, {"phase": {"func": lambda s: s == "stable", "mode": "frame-wise"}})
df_pore_filtered_de = filter_combined_df(df_pore, {"phase": {"func": lambda s: s == "decaying", "mode": "frame-wise"}})

#####################

df_base = df_spot_filtered_fo
region = "penumbra"

Ic = df_base[f"penumbra_corrected_area"]
Ic = df_base[f"Ic_{region}_corrected_border_flux_mean"]
B = df_base[f"B_{region}_corrected_border_flux_mean"]
Bhor = df_base[f"Bhor_{region}_corrected_border_flux_mean"]

data = np.vstack([Ic, B, Bhor])
kde = gaussian_kde(data)

xmin, xmax = Ic.min(), Ic.max()
ymin, ymax = B.min(), B.max()
zmin, zmax = Bhor.min(), Bhor.max()

density = kde(data)

# Plot
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

sc = ax.scatter(
    Ic, B, Bhor,
    c=density,
    cmap='viridis',
    s=10,
    alpha=0.7
)

ax.set(
    xlabel=f'penumbra_corrected_area',
    ylabel=f'B ({region})',
    zlabel=f'Bhor ({region})',
)

fig.colorbar(sc, ax=ax, label="Density")

plt.show()


#
