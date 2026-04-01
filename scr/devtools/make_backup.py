from shutil import copytree, copy2, ignore_patterns  # , make_archive
from glob import glob
from os import path

from scr.config.paths import BACKUP_DIR, PROJECT_DIR, SUBDIRS

from scr.utils.filesystem import check_dir


def make_backup(version: str) -> None:
    # definition of backup dirs and creating them
    backup_dir = path.join(BACKUP_DIR, version)

    if path.isdir(backup_dir):
        raise ValueError('The back-up dir already exists. Change "version" parameter.')

    # directories in _project_dir
    dirs_to_save_specific = {
        "./": ["*.py", "*.in", "*.toml", "*.txt", "*.sh"]
    }

    # directories in _project_dir
    dirs_to_save_all = {
        SUBDIRS["scr"],
        SUBDIRS["OpenPBS"],
    }

    # save specific suffixes
    for directory, suffixes in dirs_to_save_specific.items():
        dir_to_backup = path.join(PROJECT_DIR, directory)

        if not path.isdir(dir_to_backup):
            print(f'Directory "{dir_to_backup}" does not exist. Skipping it...')
            continue

        backup_dir_name = path.join(backup_dir, directory)
        check_dir(backup_dir_name)

        for suffix in suffixes:
            source = path.join(dir_to_backup, suffix)

            for file in glob(source):
                copy2(file, backup_dir_name)

    # save all that is inside
    for directory in dirs_to_save_all:
        source = path.join(PROJECT_DIR, directory)

        if not path.isdir(source):
            print(f'Directory "{source}" does not exist. Skipping it...')
            continue

        copytree(
            source,
            path.join(backup_dir, directory),
            ignore=ignore_patterns("__pycache__", "*.pyc")
        )

    # zip the folder
    # make_archive(backup_dir, "zip", backup_dir)


if __name__ == "__main__":
    from datetime import datetime

    make_backup(
        version=path.join(datetime.utcnow().strftime("%Y-%m-%d"), "codes")
    )
