from os import path

PROJECT_DIR = "/nfshome/david/Contours"
BACKUP_DIR = "/nfsscratch/david/backup/Contours"

PATH_TRACKS = "/nfsscratch/david/Contours/tracks"
PATH_SUNSPOTS = "/nfsscratch/david/Contours/sunspots"
PATH_SUNSPOTS_STATS = "/nfsscratch/david/Contours/sunspots_stats"
PATH_SUNSPOTS_PHASES = "/nfsscratch/david/Contours/sunspots_phases"

# Slopes defining the sunspot evolution. Good to precompute.
SLOPES_BASENAME = "flux_slopes.parquet"

SUBDIRS = {
    "scr": "scr",
    "OpenPBS": "OpenPBS",
}

PATH_SCRIPTS = path.join(PROJECT_DIR, SUBDIRS["scr"])
PATH_PBS = path.join(PROJECT_DIR, SUBDIRS["OpenPBS"])

# base folder for all generated outputs
PATH_GRAPHIC_OUTPUT = path.join(PROJECT_DIR, "graphic_output")

# subfolders for different types
PATH_FIGURES = path.join(PATH_GRAPHIC_OUTPUT, "figures")
PATH_VIDEOS = path.join(PATH_GRAPHIC_OUTPUT, "videos")
PATH_INTERACTIVE = path.join(PATH_GRAPHIC_OUTPUT, "interactive")
