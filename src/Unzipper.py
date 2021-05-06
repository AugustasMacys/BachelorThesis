import logging
import multiprocessing
import os
from time import time


from pathlib import Path
from typing import Union
from zipfile import ZipFile

from src.Utilities import TRAIN_DIRECTORY


def extract_zip(zipfile: Union[str, Path]) -> None:
    with ZipFile(zipfile) as zip_file:
        for file in zip_file.namelist():
            zip_file.extract(member=file, path=zipfile.parent)
    logging.info(f'Finished extracting {zipfile.stem}')
    os.remove(zipfile)


if __name__ == "__main__":
    DATA = TRAIN_DIRECTORY
    zipfiles = sorted(list(DATA.glob("**/*.zip")), key=lambda x: x.stem)
    logging.basicConfig(filename="extract2.log", level=logging.INFO)

    start = time()
    with multiprocessing.Pool() as pool:
        pool.map(extract_zip, zipfiles)

    logging.info(f"Extracted all zip files in {time() - start} seconds!")
