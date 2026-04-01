from glob import glob
from scr.io.sunspots import load_sunspot_file
from scr.io.tracks import load_track_file
from scr.io.fits.stack import LazyImageStack

file = sorted(glob("/nfsscratch/david/Contours/sunspots/*"))[0]

sunspots, _, metadata, _ = load_sunspot_file(file)
outer_tracks, _, _, _ = load_track_file(file.replace("sunspots", "tracks"))

images = LazyImageStack(metadata["image_paths"], quantity="Ic")

from scr.plotting.debug.plot_sunspots import plot_image_with_sunspots
from scr.statistics.computation.regions import prepare_level_data, build_regions
from scr.utils.collections import nested_defaultdict
from scr.utils.filesystem import is_empty

frame_to_sids = nested_defaultdict(factory=list, depth=1)
for sid, sunspot in sunspots.items():
    # Get all frames across all levels for this specific sunspot
    active_frames = set().union(*[sunspot[lvl].keys() for lvl in sunspot])
    for frame in active_frames:
        frame_to_sids[frame].append(sid)

from scr.devtools.plotting import plot_me

regions = ["Ic<0.9", "Ic<0.65", "Ic<0.5"]
frame = 295

image = images[frame]
sunspot_merged = nested_defaultdict(factory=list, depth=2)
from scr.geometry.contours.shapes import contour_to_shape, Point
for sid in frame_to_sids[frame]:
    spot = sunspots[sid]
    for region in regions:
        contours = spot.get(region, {}).get(frame, [])
        for contour in contours:
            sunspot_merged[region][frame].extend(contours)

level_data = prepare_level_data(
    sunspot=sunspot_merged,
    frame=frame,
    shape=image.shape,
    max_vertex_spacing=0.5,
)

regions, outermost_key = build_regions(level_data)

for region in regions:
    if not is_empty(regions[region]["total_mask"]):
        plot_me(image)
        fig, ax = plot_me(image * regions[region]["total_mask"])
        ax.set_title(region)


import numpy as np
from copy import deepcopy
from scr.geometry.contours.extraction import find_contours
from skimage.draw.draw import disk
IM = np.zeros((512, 512))
r, c = disk((512//2, 512//2), 512//4, shape=IM.shape)
IM[r, c] = 1
plot_me(IM)

c0 = find_contours(IM, 0.5)

im = deepcopy(IM)
im[200:, 220:230] = np.nan
im[200:, 200:210] = np.nan
im[200:, 180:190] = np.nan
im[200:, 160:170] = np.nan
c1 = find_contours(im, 0.5)

#############################################

from glob import glob
from tqdm import tqdm
from scr.io.sunspots import load_sunspot_file
from scr.statistics.dataframe.flatten import flatten_spot_features_with_frame
files = sorted(glob("/nfsscratch/david/Contours/sunspots_stats/*"))

wrong = []
for i in tqdm(range(len(files))):
    file = files[i]

    all_stats: dict = {}
    for stats_path in [file]:
        _, stats, _, _ = load_sunspot_file(stats_path)
        all_stats[stats_path] = stats
    if not stats:
        continue

    combined_df = flatten_spot_features_with_frame(all_stats=all_stats)

    keys = list(combined_df.keys())
    A = any("Ic<0.5-Ic<0.65" in k for k in keys)
    B = any("Ic<0.5-Ic<0.9" in k for k in keys)
    C = any("Ic<0.65-Ic<0.9" in k for k in keys)

    if A or B or C:
        print(file)
        wrong.append(file)


wrong = ['/nfsscratch/david/Contours/sunspots_stats/sunspots_Ic-0.9_AR-11089_20100725_S23E06_Ic-B-Bp-Bt-Br-Bhor.npz',
 '/nfsscratch/david/Contours/sunspots_stats/sunspots_Ic-0.9_AR-11108_20100920_S30E36_Ic-B-Bp-Bt-Br-Bhor.npz',
 '/nfsscratch/david/Contours/sunspots_stats/sunspots_Ic-0.9_AR-11112_20101015_S18W00_Ic-B-Bp-Bt-Br-Bhor.npz',
 '/nfsscratch/david/Contours/sunspots_stats/sunspots_Ic-0.9_AR-11143_20110108_S22E16_Ic-B-Bp-Bt-Br-Bhor.npz',
 '/nfsscratch/david/Contours/sunspots_stats/sunspots_Ic-0.9_AR-11147_20110121_N24E09_Ic-B-Bp-Bt-Br-Bhor.npz',
 '/nfsscratch/david/Contours/sunspots_stats/sunspots_Ic-0.9_AR-11149_20110124_N17W32_Ic-B-Bp-Bt-Br-Bhor.npz',
 '/nfsscratch/david/Contours/sunspots_stats/sunspots_Ic-0.9_AR-11178_20110325_S13E68_Ic-B-Bp-Bt-Br-Bhor.npz',
 '/nfsscratch/david/Contours/sunspots_stats/sunspots_Ic-0.9_AR-11191_20110413_N09E71_Ic-B-Bp-Bt-Br-Bhor.npz',
 '/nfsscratch/david/Contours/sunspots_stats/sunspots_Ic-0.9_AR-11207_20110506_N24E40_Ic-B-Bp-Bt-Br-Bhor.npz',
 '/nfsscratch/david/Contours/sunspots_stats/sunspots_Ic-0.9_AR-11226_20110528_S18E74_Ic-B-Bp-Bt-Br-Bhor.npz',
 '/nfsscratch/david/Contours/sunspots_stats/sunspots_Ic-0.9_AR-11231_20110602_N09E68_Ic-B-Bp-Bt-Br-Bhor.npz',
 '/nfsscratch/david/Contours/sunspots_stats/sunspots_Ic-0.9_AR-11232_20110603_N10E61_Ic-B-Bp-Bt-Br-Bhor.npz',
 '/nfsscratch/david/Contours/sunspots_stats/sunspots_Ic-0.9_AR-11266_20110805_N18E38_Ic-B-Bp-Bt-Br-Bhor.npz',
 '/nfsscratch/david/Contours/sunspots_stats/sunspots_Ic-0.9_AR-11281_20110829_S21E65_Ic-B-Bp-Bt-Br-Bhor.npz',
 '/nfsscratch/david/Contours/sunspots_stats/sunspots_Ic-0.9_AR-13664_20240507_S20E06_Ic-B-Bp-Bt-Br-Bhor.npz',
 '/nfsscratch/david/Contours/sunspots_stats/sunspots_Ic-0.9_HARP-00043_20100604_Ic-B-Bp-Bt-Br-Bhor.npz',
 '/nfsscratch/david/Contours/sunspots_stats/sunspots_Ic-0.9_HARP-00185_20100922_Ic-B-Bp-Bt-Br-Bhor.npz',
 '/nfsscratch/david/Contours/sunspots_stats/sunspots_Ic-0.9_HARP-00211_20101014_Ic-B-Bp-Bt-Br-Bhor.npz',
 '/nfsscratch/david/Contours/sunspots_stats/sunspots_Ic-0.9_HARP-00226_20101025_Ic-B-Bp-Bt-Br-Bhor.npz',
 '/nfsscratch/david/Contours/sunspots_stats/sunspots_Ic-0.9_HARP-00377_20110216_Ic-B-Bp-Bt-Br-Bhor.npz',
 '/nfsscratch/david/Contours/sunspots_stats/sunspots_Ic-0.9_HARP-00384_20110218_Ic-B-Bp-Bt-Br-Bhor.npz',
 '/nfsscratch/david/Contours/sunspots_stats/sunspots_Ic-0.9_HARP-00393_20110303_Ic-B-Bp-Bt-Br-Bhor.npz',
 '/nfsscratch/david/Contours/sunspots_stats/sunspots_Ic-0.9_HARP-00407_20110311_Ic-B-Bp-Bt-Br-Bhor.npz',
 '/nfsscratch/david/Contours/sunspots_stats/sunspots_Ic-0.9_HARP-00421_20110317_Ic-B-Bp-Bt-Br-Bhor.npz',
 '/nfsscratch/david/Contours/sunspots_stats/sunspots_Ic-0.9_HARP-00495_20110414_Ic-B-Bp-Bt-Br-Bhor.npz',
 '/nfsscratch/david/Contours/sunspots_stats/sunspots_Ic-0.9_HARP-00504_20110419_Ic-B-Bp-Bt-Br-Bhor.npz',
 '/nfsscratch/david/Contours/sunspots_stats/sunspots_Ic-0.9_HARP-00540_20110428_Ic-B-Bp-Bt-Br-Bhor.npz',
 '/nfsscratch/david/Contours/sunspots_stats/sunspots_Ic-0.9_HARP-00556_20110505_Ic-B-Bp-Bt-Br-Bhor.npz',
 '/nfsscratch/david/Contours/sunspots_stats/sunspots_Ic-0.9_HARP-00576_20110508_Ic-B-Bp-Bt-Br-Bhor.npz',
 '/nfsscratch/david/Contours/sunspots_stats/sunspots_Ic-0.9_HARP-00650_20110606_Ic-B-Bp-Bt-Br-Bhor.npz',
 '/nfsscratch/david/Contours/sunspots_stats/sunspots_Ic-0.9_HARP-00661_20110615_Ic-B-Bp-Bt-Br-Bhor.npz',
 '/nfsscratch/david/Contours/sunspots_stats/sunspots_Ic-0.9_HARP-00798_20110821_Ic-B-Bp-Bt-Br-Bhor.npz',
 '/nfsscratch/david/Contours/sunspots_stats/sunspots_Ic-0.9_HARP-00833_20110905_Ic-B-Bp-Bt-Br-Bhor.npz',
 '/nfsscratch/david/Contours/sunspots_stats/sunspots_Ic-0.9_HARP-00843_20110909_Ic-B-Bp-Bt-Br-Bhor.npz',
 '/nfsscratch/david/Contours/sunspots_stats/sunspots_Ic-0.9_HARP-00847_20110912_Ic-B-Bp-Bt-Br-Bhor.npz',
 '/nfsscratch/david/Contours/sunspots_stats/sunspots_Ic-0.9_HARP-00892_20110928_Ic-B-Bp-Bt-Br-Bhor.npz',
 '/nfsscratch/david/Contours/sunspots_stats/sunspots_Ic-0.9_HARP-00903_20111001_Ic-B-Bp-Bt-Br-Bhor.npz',
 '/nfsscratch/david/Contours/sunspots_stats/sunspots_Ic-0.9_HARP-00932_20111011_Ic-B-Bp-Bt-Br-Bhor.npz',
 '/nfsscratch/david/Contours/sunspots_stats/sunspots_Ic-0.9_HARP-00940_20111015_Ic-B-Bp-Bt-Br-Bhor.npz',
 '/nfsscratch/david/Contours/sunspots_stats/sunspots_Ic-0.9_HARP-00950_20111016_Ic-B-Bp-Bt-Br-Bhor.npz',
 '/nfsscratch/david/Contours/sunspots_stats/sunspots_Ic-0.9_HARP-00982_20111023_Ic-B-Bp-Bt-Br-Bhor.npz',
 '/nfsscratch/david/Contours/sunspots_stats/sunspots_Ic-0.9_HARP-00997_20111028_Ic-B-Bp-Bt-Br-Bhor.npz',
 '/nfsscratch/david/Contours/sunspots_stats/sunspots_Ic-0.9_HARP-01028_20111108_Ic-B-Bp-Bt-Br-Bhor.npz',
 '/nfsscratch/david/Contours/sunspots_stats/sunspots_Ic-0.9_HARP-01124_20111205_Ic-B-Bp-Bt-Br-Bhor.npz',
 '/nfsscratch/david/Contours/sunspots_stats/sunspots_Ic-0.9_HARP-01168_20111214_Ic-B-Bp-Bt-Br-Bhor.npz',
 '/nfsscratch/david/Contours/sunspots_stats/sunspots_Ic-0.9_HARP-01256_20120102_Ic-B-Bp-Bt-Br-Bhor.npz',
 '/nfsscratch/david/Contours/sunspots_stats/sunspots_Ic-0.9_HARP-01278_20120108_Ic-B-Bp-Bt-Br-Bhor.npz',
 '/nfsscratch/david/Contours/sunspots_stats/sunspots_Ic-0.9_HARP-01350_20120201_Ic-B-Bp-Bt-Br-Bhor.npz',
 '/nfsscratch/david/Contours/sunspots_stats/sunspots_Ic-0.9_HARP-01391_20120214_Ic-B-Bp-Bt-Br-Bhor.npz',
 '/nfsscratch/david/Contours/sunspots_stats/sunspots_Ic-0.9_HARP-01447_20120307_Ic-B-Bp-Bt-Br-Bhor.npz',
 '/nfsscratch/david/Contours/sunspots_stats/sunspots_Ic-0.9_HARP-01449_20120309_Ic-B-Bp-Bt-Br-Bhor.npz',
 '/nfsscratch/david/Contours/sunspots_stats/sunspots_Ic-0.9_HARP-01471_20120317_Ic-B-Bp-Bt-Br-Bhor.npz',
 '/nfsscratch/david/Contours/sunspots_stats/sunspots_Ic-0.9_HARP-11149_20240507_Ic-B-Bp-Bt-Br-Bhor.npz']










import numpy as np
from scr.io.sunspots import load_sunspot_file
from scr.io.tracks import load_track_file
from scr.io.fits.stack import LazyImageStack
from scr.statistics.dataframe.flatten import flatten_spot_features_with_frame
from scr.plotting.debug.plot_sunspots import plot_image_with_sunspots
from scr.plotting.debug.plot_tracks import plot_image_with_tracks
import matplotlib
matplotlib.use("TkAgg")

all_stats: dict = {}
for stats_path in wrong[:5]:
    _, stats, _, _ = load_sunspot_file(stats_path)
    all_stats[stats_path] = stats

combined_df = flatten_spot_features_with_frame(all_stats=all_stats)

keys = list(combined_df.keys())

for corupted_type in ["Ic<0.5-Ic<0.65", "Ic<0.5-Ic<0.9", "Ic<0.65-Ic<0.9"]:
    corupted = [k for k in keys if corupted_type in k]
    if not corupted:
        continue
    print(corupted)

    bad_iloc = np.where(np.isfinite(combined_df[corupted[0]]))[0]
    df = combined_df.iloc[bad_iloc]

    for i in range(len(df)):
        series = df.iloc[i]
        sunspots, _, metadata, _ = load_sunspot_file(series["observation_id"])
        frame = series["frame"]
        image_paths = metadata["image_paths"]
        images = LazyImageStack(image_paths, "Ic")
        image = images[frame]

        plot_image_with_sunspots(image=image, sunspots=sunspots, frame=frame)

tracks, _, metadata, _ = load_track_file(("_".join(series["observation_id"].replace("sunspots", "tracks").split("_")[:-1])+".npz").replace("_stats", ""))
tracks = tracks["Ic<0.9"]
plot_image_with_tracks(image, tracks, frame=frame)

outer_contours = tracks[0][frame]


standard:


filled:
 '/nfsscratch/david/Contours/tracks/tracks_Ic-0.9_HARP-00940_20111015.npz',
 '/nfsscratch/david/Contours/tracks/tracks_Ic-0.9_HARP-01126_20111207.npz'
tracks_Ic-0.9_HARP-01256_20120102.npz
tracks_Ic-0.9_HARP-00514_20110425.npz


start = time()

stop = time()
print(f"XYZ: {stop - start} seconds")